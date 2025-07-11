import json
import requests
import time
import asyncio
import aiohttp
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from config import Config

class APITester:
    def __init__(self, api_config: Dict[str, Any] = None):
        if api_config is None:
            api_config = Config.get_api_config()
        
        self.base_url = api_config["base_url"].rstrip("/")
        self.query_endpoint = api_config.get("query_endpoint", "/query/{question}")
        self.timeout = api_config.get("timeout", 30)
        self.delay = api_config.get("delay", 1)
        
        self.results_dir = Path(Config.RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)
    
    def query_rag_api(self, question: str) -> Dict[str, Any]:
        if "{question}" in self.query_endpoint:
            url = f"{self.base_url}{self.query_endpoint.format(question=question)}"
        else:
            url = f"{self.base_url}{self.query_endpoint}"
        
        try:
            start_time = time.time()
            
            if "{question}" in self.query_endpoint:
                response = requests.get(url, timeout=self.timeout)
            else:
                # Format for LangServe API
                payload = {
                    "input": {
                        "input": question,
                        "next": "string",
                        "chat_history": [],
                        "extra": {},
                        "user_intent": {}
                    },
                    "config": {
                        "configurable": {
                            "checkpoint_id": "string",
                            "checkpoint_ns": "",
                            "thread_id": ""
                        }
                    },
                    "kwargs": {}
                }
                response = requests.post(url, json=payload, timeout=self.timeout)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer = self._extract_answer_from_response(result)
                
                return {
                    "status": "success",
                    "question": question,
                    "answer": answer,
                    "full_response": result,
                    "response_time": response_time,
                    "status_code": response.status_code
                }
            else:
                return {
                    "status": "error",
                    "question": question,
                    "answer": f"API Error: {response.status_code}",
                    "full_response": response.text,
                    "response_time": response_time,
                    "status_code": response.status_code
                }
                
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "question": question,
                "answer": "Request timeout",
                "full_response": None,
                "response_time": self.timeout,
                "status_code": None
            }
        except Exception as e:
            return {
                "status": "error",
                "question": question,
                "answer": f"Error: {str(e)}",
                "full_response": None,
                "response_time": None,
                "status_code": None
            }
    
    def _extract_answer_from_response(self, response: Dict[str, Any]) -> str:
        # common response formats to try
        possible_paths = [
            ["output", "output", "content"],   # LangServe format
            ["output", "content"],            # Alternative LangServe format
            ["response", "output"],           # dugbot format
            ["answer"],
            ["result"],
            ["output"],
            ["text"],
            ["content"]
        ]
        
        for path in possible_paths:
            try:
                current = response
                for key in path:
                    current = current[key]
                if isinstance(current, str) and current.strip():
                    return current.strip()
            except (KeyError, TypeError):
                continue
        
        return str(response)
    
    def test_dataset(self, dataset: Dict[str, Any], output_prefix: str = "api_test") -> Dict[str, Any]:
        print(f"Testing API with {len(dataset['questions'])} questions...")
        
        results = []
        successful_requests = 0
        failed_requests = 0
        total_response_time = 0
        
        for i, question_data in enumerate(dataset['questions']):
            question = question_data.get('question', question_data.get('user_input', ''))
            expected_answer = question_data.get('expected_answer', question_data.get('reference', ''))
            
            print(f"Processing {i+1}/{len(dataset['questions'])}: {question[:80]}...")
            
            api_result = self.query_rag_api(question)
            
            if api_result['status'] == 'success':
                successful_requests += 1
                total_response_time += api_result['response_time']
            else:
                failed_requests += 1
            
            result = {
                "question_id": question_data.get('question_id', f"Q{i+1:03d}"),
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": api_result['answer'],
                "api_status": api_result['status'],
                "response_time": api_result['response_time'],
                "status_code": api_result['status_code']
            }
            results.append(result)
            
            if self.delay > 0:
                time.sleep(self.delay)
        
        avg_response_time = total_response_time / successful_requests if successful_requests > 0 else 0
        
        evaluation_summary = {
            "test_info": {
                "total_questions": len(results),
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / len(results) if len(results) > 0 else 0,
                "average_response_time": avg_response_time,
                "test_date": datetime.now().isoformat()
            },
            "results": results
        }
        
        output_file = self.results_dir / f"{output_prefix}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Testing completed!")
        print(f"Success rate: {successful_requests}/{len(results)} ({evaluation_summary['test_info']['success_rate']:.2%})")
        print(f"Average response time: {avg_response_time:.2f}s")
        print(f"Results saved to: {output_file}")
        
        return evaluation_summary
    
    def generate_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        test_info = results['test_info']
        
        report = []
        report.append("API Testing Report")
        report.append("=" * 40)
        report.append("")
        report.append(f"Test Date: {test_info['test_date']}")
        report.append(f"Total Questions: {test_info['total_questions']}")
        report.append(f"Successful Requests: {test_info['successful_requests']}")
        report.append(f"Failed Requests: {test_info['failed_requests']}")
        report.append(f"Success Rate: {test_info['success_rate']:.2%}")
        report.append(f"Average Response Time: {test_info['average_response_time']:.2f}s")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        
        return report_text