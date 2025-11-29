"""
智能 Agent 分析模块：使用 LLM 对高风险文本对进行深度分析与解释。
包括证据采样、上下文提取、控辩双方视角推理与综合判定。
支持 provider 抽象（OpenAI 或通用 HTTP），默认兼容现有配置。
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import requests
from openai import OpenAI


@dataclass
class AgentAnalysis:
    """Agent分析结果"""
    is_plagiarism: bool
    confidence: float  # 0-1
    reasoning: str
    key_evidence: List[str]
    defense_points: List[str]


class SmartPlagiarismAgent:
    """
    智能抄袭分析 Agent。

    读取 API 配置，初始化客户端与模型；提供方向性对的分析能力。
    """
    
    def __init__(self, api_config_path: str = "api_config.json", dual_phase: bool = True):
        """
        初始化 Agent。

        Args:
            api_config_path: API 配置文件路径，需包含 base_url/api_key/model。
        """
        self.config = self._load_config(api_config_path)
        self.model = self.config.get('model')
        self.provider = self.config.get('provider', 'openai')
        self.dual_phase = dual_phase
        if self.provider == 'openai':
            self.client = OpenAI(
                base_url=self.config['base_url'],
                api_key=self.config['api_key']
            )
        else:
            self.client = None
    
    def _load_config(self, config_path: str) -> Dict:
        """
        加载 API 配置。

        Args:
            config_path: 配置文件路径。

        Returns:
            解析后的配置字典。

        Raises:
            FileNotFoundError: 配置文件不存在。
            KeyError: 配置中缺少必须字段。
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"配置文件 {config_path} 不存在！请创建并填入API key"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 兼容原有 'modelscope' 字段；也支持 'openai' 或通用 'provider' 字段
        if 'modelscope' in config:
            cfg = config['modelscope']
            cfg.setdefault('provider', 'openai')
            return cfg
        if 'openai' in config:
            cfg = config['openai']
            cfg['provider'] = 'openai'
            return cfg
        if 'provider' in config:
            return config
        raise KeyError("配置文件格式错误，缺少 'modelscope'/'openai'/'provider' 字段")
    
    def analyze_suspicious_pair(
        self,
        text_a: str,
        text_b: str,
        similarity_hits: List[Dict],
        statistics: Dict,
        left_name: str,
        right_name: str,
        dual_phase: bool = True
    ) -> AgentAnalysis:
        """
        分析方向性文本对（左=Target，右=Reference），支持单/双阶段调用。
        中文注释：统一组织证据与统计，并传递明确的角色名，避免 A/B 混淆。
        """
        # 智能采样证据
        top_evidence = self._select_representative_evidence(similarity_hits, max_samples=5)
        
        # 提取上下文
        evidence_contexts = self._extract_contexts(top_evidence)
        
        # 检察官视角分析（中文注释：加入 Target/Reference 名称，提高可读性）
        prosecutor_prompt = self._build_prosecutor_prompt(evidence_contexts, statistics, left_name, right_name)
        prosecutor_result = self._call_llm(prosecutor_prompt)
        defense_result = {'defense_points': []}
        if dual_phase and self.dual_phase:
            defense_prompt = self._build_defense_prompt(evidence_contexts, prosecutor_result)
            defense_result = self._call_llm(defense_prompt)
        
        # 综合判断
        final_analysis = self._synthesize_judgment(
            prosecutor_result, 
            defense_result, 
            statistics
        )
        
        return final_analysis
    
    def _select_representative_evidence(
        self, 
        hits: List[Dict], 
        max_samples: int = 5
    ) -> List[Dict]:
        """
        智能采样：从命中中选择最具代表性的证据集合。

        策略：最高相似度、若干中等相似度、位置分散的样本。
        """
        if len(hits) <= max_samples:
            return hits
        
        sorted_hits = sorted(hits, key=lambda x: x['sim'], reverse=True)
        selected = []
        
        # 最高相似度1对
        selected.append(sorted_hits[0])
        
        # 中等相似度2对
        mid_start = len(sorted_hits) // 3
        mid_end = 2 * len(sorted_hits) // 3
        mid_range = sorted_hits[mid_start:mid_end]
        if len(mid_range) >= 2:
            step = len(mid_range) // 2
            selected.extend([mid_range[0], mid_range[step]])
        
        # 位置分散2对
        remaining = [h for h in sorted_hits if h not in selected]
        if len(remaining) >= 2:
            positions = [h['sent_id_i'] for h in remaining]
            min_pos_idx = positions.index(min(positions))
            max_pos_idx = positions.index(max(positions))
            selected.extend([remaining[min_pos_idx], remaining[max_pos_idx]])
        
        return selected[:max_samples]
    
    def _extract_contexts(
        self,
        evidence: List[Dict]
    ) -> List[Dict]:
        """
        提取证据上下文信息（文本片段、相似度、位置、是否含引用）。
        """
        contexts = []
        for hit in evidence:
            context = {
                'text_a': hit['text_i'],
                'text_b': hit['text_j'],
                'similarity': hit['sim'],
                'position_a': hit['sent_id_i'],
                'position_b': hit['sent_id_j'],
                'has_citation': hit.get('citation_penalty', 1.0) < 1.0
            }
            contexts.append(context)
        return contexts
    
    def _build_prosecutor_prompt(self, evidence: List[Dict], stats: Dict, left_name: str, right_name: str) -> str:
        """Build prosecutor perspective prompt (returns JSON). Clear Target/Reference names to avoid A/B confusion."""
        prompt = f"""You are an academic integrity prosecutor analyzing potential plagiarism.

**Statistical Data:**
- Similar sentences: {stats.get('count', 0)}
- Average similarity: {stats.get('mean_sim', 0):.1%}
- Maximum similarity: {stats.get('max_sim', 0):.1%}
- Text coverage: {stats.get('coverage_min', 0):.1%}

**Roles:**
- Target (under review): {left_name}
- Reference (source): {right_name}

**Key Evidence ({len(evidence)} pairs):**
"""
        
        for i, ctx in enumerate(evidence, 1):
            citation_mark = " [with citation]" if ctx['has_citation'] else ""
            prompt += f"""
Evidence {i} (similarity {ctx['similarity']:.1%}){citation_mark}
  Target({left_name}): "{ctx['text_a']}"
  Reference({right_name}): "{ctx['text_b']}"
"""
        
        prompt += """
**Please analyze (must return valid JSON):**
{
  "is_plagiarism": true,
  "confidence": 85,
  "reasoning": "Based on evidence...",
  "key_evidence": ["Evidence 1...", "Evidence 2..."],
  "evidence_assessments": [
    {"type":"verbatim","idx":1},
    {"type":"semantic","idx":2},
    {"type":"common_knowledge","idx":3},
    {"type":"quoted_with_citation","idx":4}
  ],
  "cross_lingual_mapping": true,
  "style_shift": false
}
Example:
{
  "is_plagiarism": false,
  "confidence": 60,
  "reasoning": "Sufficient citations and low coverage",
  "key_evidence": ["Evidence 1"],
  "evidence_assessments": [{"type":"common_knowledge","idx":1}],
  "cross_lingual_mapping": false,
  "style_shift": false
}
"""
        return prompt
    
    def _build_defense_prompt(self, evidence: List[Dict], prosecutor: Dict) -> str:
        """Build defense lawyer perspective prompt (requires JSON return)."""
        reasoning = prosecutor.get('reasoning', 'Unknown')
        
        prompt = f"""You are an academic defense attorney. The prosecution claims plagiarism with reasoning: {reasoning}

**Evidence:**
"""
        for i, ctx in enumerate(evidence, 1):
            citation = " [with citation]" if ctx['has_citation'] else ""
            prompt += f"{i}. Similarity {ctx['similarity']:.1%}{citation}\n"
        
        prompt += """
**Please provide defense (must return valid JSON):**
{
  "defense_points": ["Reason 1", "Reason 2"],
  "weakness": "The prosecution overlooked...",
  "alternative_explanation": "This could be..."
}
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> Dict:
        """
        调用 LLM API 并解析（尽量提取 JSON）。

        Returns:
            Dict: 若解析成功返回JSON字典，否则包含原始响应或错误信息。
        """
        print(f"\n      🌐 _call_llm invoked")
        print(f"         Provider: {self.provider}")
        print(f"         Model: {self.model}")
        print(f"         Prompt length: {len(prompt)} chars")
        
        try:
            full_response = ""
            if self.provider == 'openai' and self.client is not None:
                print(f"         Using OpenAI-compatible client...")
                # 非流式，增强兼容性
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.3
                )
                full_response = response.choices[0].message.content or ""
                print(f"         ✅ Response received: {len(full_response)} chars")
            else:
                print(f"         Using generic HTTP provider...")
                # 通用 HTTP provider：POST 到 base_url
                base_url = self.config.get('base_url')
                api_key = self.config.get('api_key')
                print(f"         API URL: {base_url}")
                headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
                payload = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.3
                }
                resp = requests.post(base_url, headers=headers, json=payload, timeout=30)
                print(f"         HTTP Status: {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                if 'choices' in data:
                    full_response = data['choices'][0]['message']['content']
                else:
                    full_response = json.dumps(data, ensure_ascii=False)
                print(f"         ✅ Response received: {len(full_response)} chars")

            # 解析 JSON
            print(f"         🔍 Parsing JSON from response...")
            json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                print(f"         ✅ JSON parsed successfully")
                print(f"         Keys in parsed JSON: {list(parsed.keys())}")
                return parsed
            else:
                print(f"         ⚠️ No JSON found in response")
                print(f"         First 200 chars of response: {full_response[:200]}")
                return {'raw_response': full_response, 'parse_error': True}
        except Exception as e:
            print(f"         ❌ LLM call failed")
            print(f"         Error type: {type(e).__name__}")
            print(f"         Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'error_type': type(e).__name__}
    
    def _synthesize_judgment(
        self,
        prosecutor: Dict,
        defense: Dict,
        stats: Dict
    ) -> AgentAnalysis:
        """
        综合控辩双方输出与统计，生成最终判定。
        """
        if 'error' in prosecutor:
            return AgentAnalysis(
                is_plagiarism=False,
                confidence=0.0,
                reasoning=f"Analysis error: {prosecutor['error']}",
                key_evidence=[],
                defense_points=[]
            )
        
        is_plagiarism = prosecutor.get('is_plagiarism', False)
        confidence = prosecutor.get('confidence', 50) / 100.0
        
        # 辩护调整
        defense_points = defense.get('defense_points', [])
        if defense_points:
            confidence_penalty = min(0.3, len(defense_points) * 0.1)
            confidence *= (1 - confidence_penalty)
        
        # 统计校准
        coverage = stats.get('coverage_min', 0)
        mean_sim = stats.get('mean_sim', 0)
        if coverage > 0.8 and mean_sim > 0.9:
            confidence = min(0.95, confidence * 1.2)
        
        # 引用校准
        avg_citation_penalty = stats.get('avg_citation_penalty', 1.0)
        if avg_citation_penalty < 0.5:
            confidence *= 0.7
            is_plagiarism = False
        
        return AgentAnalysis(
            is_plagiarism=is_plagiarism,
            confidence=confidence,
            reasoning=prosecutor.get('reasoning', ''),
            key_evidence=prosecutor.get('key_evidence', []),
            defense_points=defense_points
        )


def generate_agent_report(
    agent: SmartPlagiarismAgent,
    pair_detail: Dict,
    text_a: str,
    text_b: str,
    dual_phase: bool = True
) -> str:
    """
    生成可视化报告文案（Markdown）。
    """
    # 中文注释：传入明确的文档名，避免 A/B 混淆
    left_id = pair_detail['pair'][0]
    right_id = pair_detail['pair'][1]
    analysis = agent.analyze_suspicious_pair(
        text_a=text_a,
        text_b=text_b,
        similarity_hits=pair_detail['hits'],
        statistics={
            'count': pair_detail['count'],
            'mean_sim': pair_detail['mean_sim'],
            'max_sim': pair_detail['max_sim'],
            'coverage_min': pair_detail['coverage_min']
        },
        left_name=left_id,
        right_name=right_id,
        dual_phase=dual_phase
    )
    
    status_emoji = "⚠️" if analysis.is_plagiarism else "✅"
    status_text = "Potential plagiarism detected" if analysis.is_plagiarism else "No obvious plagiarism detected"
    
    report = f"""
## {status_emoji} AI Analysis Results

**Verdict**: {status_text}  
**Confidence**: {analysis.confidence:.1%}

### 🤔 AI Reasoning Process
{analysis.reasoning}

### 📌 Key Evidence
"""
    for i, evidence in enumerate(analysis.key_evidence, 1):
        report += f"{i}. {evidence}\n"
    
    if analysis.defense_points:
        report += "\n### 🛡️ Possible Defense Arguments\n"
        for point in analysis.defense_points:
            report += f"- {point}\n"
    report += "\n### 🧭 Semantic and Style Assessment\n- Cross-lingual semantic mapping: Please refer to reasoning above\n- Writing style variation: Please refer to reasoning above\n"
    # System-generated segment list (Top5) with clear Target/Reference and similarity scores.
    hits = pair_detail.get('hits', [])[:5]
    if hits:
        report += "\n### 📑 Suspicious Segments (Top 5)\n"
        for i, h in enumerate(hits, 1):
            ta = h['text_i']
            tb = h['text_j']
            sim = h.get('adjusted_sim', h['sim'])
            report += f"{i}. Target({left_id}): \"{ta}\"\n   Reference({right_id}): \"{tb}\"\n   Similarity: {sim:.2f}\n"
    
    report += f"""
### 📊 Statistical Summary
- Similar sentences: {pair_detail['count']}
- Average similarity: {pair_detail['mean_sim']:.1%}
- Maximum similarity: {pair_detail['max_sim']:.1%}
- Text coverage: {pair_detail['coverage_min']:.1%}
"""
    
    return report

def generate_agent_report_batch(
    agent: SmartPlagiarismAgent,
    details: List[Dict],
    texts: Dict[str, str],
    dual_phase: bool = False
) -> List[Dict]:
    print(f"\n🤖 generate_agent_report_batch called")
    print(f"   Details count: {len(details)}")
    print(f"   Texts keys: {list(texts.keys())}")
    print(f"   Dual phase: {dual_phase}")
    
    items = []
    for idx, d in enumerate(details, 1):
        pair = d['pair']
        hits = d.get('hits', [])
        print(f"\n   Processing pair {idx}/{len(details)}: {pair[0]} vs {pair[1]}")
        print(f"      Hits count: {len(hits)}")
        print(f"      Stats: count={d.get('count')}, mean_sim={d.get('mean_sim', 0):.3f}")
        
        top_hits = sorted(hits, key=lambda x: x['sim'], reverse=True)[:5]
        evidence = []
        for h in top_hits:
            evidence.append({
                'text_a': h['text_i'][:160],
                'text_b': h['text_j'][:160],
                'similarity': h['sim'],
                'position_a': h['sent_id_i'],
                'position_b': h['sent_id_j'],
                'has_citation': h.get('citation_penalty', 1.0) < 1.0
            })
        
        full_a = texts.get(pair[0], '')
        full_b = texts.get(pair[1], '')
        print(f"      Full text A length: {len(full_a)} chars (using {min(len(full_a), 8000)} chars)")
        print(f"      Full text B length: {len(full_b)} chars (using {min(len(full_b), 8000)} chars)")
        print(f"      Evidence samples: {len(evidence)}")
        
        items.append({
            'pair': pair,
            'stats': {
                'count': d['count'],
                'mean_sim': d['mean_sim'],
                'max_sim': d['max_sim'],
                'coverage_min': d['coverage_min']
            },
            'evidence': evidence,
            'full_a': full_a[:8000],
            'full_b': full_b[:8000]
        })
    print(f"\n   📝 Building prompt for {len(items)} pairs...")
    prompt = "Batch analyze the following directional text pairs, return JSON list reports." + "\n"
    for it in items:
        prompt += f"PAIR Target({it['pair'][0]}) -> Reference({it['pair'][1]})\n"
        s = it['stats']
        prompt += f"count={s['count']}, mean={s['mean_sim']:.2f}, max={s['max_sim']:.2f}, coverage={s['coverage_min']:.2f}\n"
        prompt += f"FULL_A: {it['full_a']}\n"
        prompt += f"FULL_B: {it['full_b']}\n"
        for i, e in enumerate(it['evidence'], 1):
            mark = " [with citation]" if e['has_citation'] else ""
            prompt += f"E{i}{mark}: A=\"{e['text_a']}\" B=\"{e['text_b']}\" sim={e['similarity']:.2f}\n"
    prompt += "\nReturn in the following format:\n{" + "\n  \"reports\": [" + "\n    {\"pair\":[\"A\",\"B\"],\"is_plagiarism\":true,\"confidence\":0.85,\"reasoning\":\"...\",\"key_evidence\":[\"...\"],\"defense_points\":[\"...\"]}" + "\n  ]\n}"
    
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   🌐 Calling LLM API (model: {agent.model})...")
    
    try:
        res = agent._call_llm(prompt)
        print(f"   ✅ LLM response received")
        print(f"   Response type: {type(res)}")
        if isinstance(res, dict):
            print(f"   Response keys: {list(res.keys())}")
    except Exception as e:
        print(f"   ❌ LLM call failed: {e}")
        raise
    reports = []
    print(f"\n   📊 Processing LLM response...")
    if isinstance(res, dict) and 'reports' in res:
        print(f"   Found 'reports' in response, count: {len(res['reports'])}")
        for idx, r in enumerate(res['reports'], 1):
            pair = r.get('pair')
            print(f"\n   Building report {idx} for pair: {pair}")
            print(f"      Is plagiarism: {r.get('is_plagiarism')}")
            print(f"      Confidence: {r.get('confidence', 0)}")
            
            md = f"## {'⚠️' if r.get('is_plagiarism') else '✅'} AI Analysis Results\n\n"
            md += f"**Verdict**: {'Potential plagiarism detected' if r.get('is_plagiarism') else 'No obvious plagiarism detected'}  \n"
            md += f"**Confidence**: {float(r.get('confidence',0))*100:.1f}%\n\n"
    else:
        print(f"   ⚠️ Unexpected response format - missing 'reports' key")
        if isinstance(res, dict):
            print(f"   Available keys: {list(res.keys())}")
            md += "### 🤔 AI Reasoning Process\n" + str(r.get('reasoning','')) + "\n\n"
            md += "### 📌 Key Evidence\n"
            for i, ev in enumerate(r.get('key_evidence', []), 1):
                md += f"{i}. {ev}\n"
            if r.get('defense_points'):
                md += "\n### 🛡️ Possible Defense Arguments\n"
                for p in r['defense_points']:
                    md += f"- {p}\n"
            reports.append({'pair': pair, 'report': md})
        return reports
    fallback = []
    for d in details:
        a = texts.get(d['pair'][0], '')
        b = texts.get(d['pair'][1], '')
        fallback.append({'pair': d['pair'], 'report': generate_agent_report(agent, d, a, b, dual_phase=dual_phase)})
    return fallback
