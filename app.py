"""
抄袭检测系统（Web界面）：文件上传、参数配置、运行检测、结果展示与导出。
支持目标文件模式与全文件比较、Agent深度分析与引用识别可视化。
"""

import streamlit as st
from pathlib import Path
import shutil
import tempfile
import re
import json

from plagiarism_checker.pipeline import PipelineConfig, PlagiarismPipeline


st.set_page_config(
    page_title="Plagiarism Detection System",
    page_icon="🔍",
    layout="wide",
)

# CSS Styles
st.markdown("""
<style>
    .highlight-high {
        background-color: #ff6b6b;
        padding: 2px 4px;
        border-radius: 3px;
        cursor: pointer;
        display: inline-block;
        margin: 2px 0;
    }
    .highlight-medium {
        background-color: #ffd93d;
        padding: 2px 4px;
        border-radius: 3px;
        cursor: pointer;
        display: inline-block;
        margin: 2px 0;
    }
    .highlight-low {
        background-color: #a8e6cf;
        padding: 2px 4px;
        border-radius: 3px;
        cursor: pointer;
        display: inline-block;
        margin: 2px 0;
    }
    .highlight-citation {
        background-color: #d4a5ff;
        padding: 2px 4px;
        border-radius: 3px;
        cursor: pointer;
        display: inline-block;
        margin: 2px 0;
        border: 1px dashed #9d4edd;
    }
    .text-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        line-height: 2;
        font-size: 16px;
        max-height: 600px;
        overflow-y: auto;
    }
    .student-name {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .target-file {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .reference-file {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .legend-item {
        display: inline-block;
        margin-right: 20px;
        margin-bottom: 10px;
    }
    .legend-box {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 3px;
        margin-right: 5px;
        vertical-align: middle;
    }
    .agent-report {
        background-color: #f0f7ff;
        border: 2px solid #4a90e2;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'selected_pair' not in st.session_state:
    st.session_state.selected_pair = None
if 'target_file' not in st.session_state:
    st.session_state.target_file = None
if 'target_files' not in st.session_state:
    st.session_state.target_files = []
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = "all"


def cleanup_temp():
    """清理临时目录与文件。"""
    if st.session_state.temp_dir and Path(st.session_state.temp_dir).exists():
        shutil.rmtree(st.session_state.temp_dir)
        st.session_state.temp_dir = None


def save_uploaded_files(target_files, reference_files):
    cleanup_temp()
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = temp_dir
    for tf in target_files:
        tpath = Path(temp_dir) / tf.name
        with open(tpath, 'wb') as f:
            f.write(tf.getbuffer())
    for ref_file in reference_files:
        rpath = Path(temp_dir) / ref_file.name
        with open(rpath, 'wb') as f:
            f.write(ref_file.getbuffer())
    return temp_dir


def save_all_files(uploaded_files):
    """保存上传的所有文件（全比较模式）。"""
    cleanup_temp()
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = temp_dir
    
    for uploaded_file in uploaded_files:
        file_path = Path(temp_dir) / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir


def run_detection(submissions_dir, config_params):
    """运行检测流程（可选Agent分析），并写出报告文件。"""
    # 提取 Agent 配置参数
    enable_agent = config_params.get('enable_agent', False)
    agent_threshold = config_params.get('agent_threshold', 0.5)
    agent_max_reports = config_params.get('agent_max_reports', 3)
    agent_dual_phase = config_params.get('agent_dual_phase', False)
    
    # 打印配置信息用于调试
    print("\n" + "="*60)
    print("🔧 Detection Configuration")
    print("="*60)
    print(f"📁 Submissions Dir: {submissions_dir}")
    print(f"🎯 Similarity Threshold: {config_params['threshold']}")
    print(f"🤖 Agent Enabled: {enable_agent}")
    if enable_agent:
        print(f"   ├─ Agent Threshold: {agent_threshold}")
        print(f"   ├─ Max Reports: {agent_max_reports}")
        print(f"   ├─ Dual Phase: {agent_dual_phase}")
        print(f"   └─ API Config: {config_params.get('api_config_path', 'api_config.json')}")
    print("="*60 + "\n")
    
    config = PipelineConfig(
        submissions_dir=Path(submissions_dir),
        output_dir=Path(submissions_dir),
        device=config_params['device'],
        use_parallel=config_params['use_parallel'],
        num_workers=config_params['num_workers'],
        similarity_threshold=config_params['threshold'],
        enable_paragraph_check=config_params['enable_paragraph'],
        enable_citation_check=config_params['enable_citation'],
        enable_multilingual=config_params['enable_multilingual'],
        para_threshold=config_params['para_threshold'],
        enable_agent=enable_agent,
        agent_threshold=agent_threshold,
        api_config_path=config_params.get('api_config_path', 'api_config.json'),
        index_top_k=config_params.get('index_top_k', 10),
        para_top_k=config_params.get('para_top_k', 5),
        agent_max_reports=agent_max_reports,
        agent_dual_phase=agent_dual_phase,
        target_stems=config_params.get('target_stems'),
        reference_stems=config_params.get('reference_stems'),
    )
    
    pipeline = PlagiarismPipeline(config)
    
    with st.spinner('Analyzing text similarity...'):
        if config.enable_agent:
            print(f"\n🤖 Starting Agent Analysis...")
            print(f"   Agent Threshold: {config.agent_threshold}")
            print(f"   Max Reports: {config.agent_max_reports}")
            try:
                sent_stats, sent_details, agent_reports = pipeline.run_with_agent()
                print(f"✅ Agent analysis completed: {len(agent_reports)} reports generated")
                para_stats, para_details = [], []
                
                # 显示候选统计
                candidates_count = sum(1 for d in sent_details if d.get('score', 0) >= config.agent_threshold)
                print(f"📊 Candidate pairs (score >= {config.agent_threshold}): {candidates_count}")
                print(f"📝 Actual reports generated: {len(agent_reports)}")
                
            except Exception as e:
                import traceback
                print(f"\n❌ Agent analysis failed with error:")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {str(e)}")
                print(f"\n📋 Full traceback:")
                traceback.print_exc()
                print(f"\n⚠️ Falling back to normal detection...\n")
                # 如果Agent失败，继续正常检测
                sent_stats, sent_details = pipeline.run()
                agent_reports = []
                para_stats, para_details = [], []
        else:
            print(f"\n🔍 Running standard detection (Agent disabled)...")
            if config.enable_paragraph_check:
                sent_stats, sent_details, para_stats, para_details = pipeline.run_with_paragraphs()
            else:
                sent_stats, sent_details = pipeline.run()
                para_stats, para_details = [], []
            agent_reports = []
        
        # 显示统计信息
        if sent_stats:
            high_risk_count = sum(1 for s in sent_stats if s['score'] >= config.agent_threshold)
            print(f"📈 Detection stats: {len(sent_stats)} pairs, {high_risk_count} high-risk pairs (≥{config.agent_threshold})")
        
        pipeline.write_reports(sent_stats, sent_details, para_stats, para_details)
        return {
            'sent_stats': sent_stats,
            'sent_details': sent_details,
            'para_stats': para_stats,
            'para_details': para_details,
            'agent_reports': agent_reports,
        }


def filter_results_by_targets(results, target_filenames):
    """过滤结果，只保留目标文件相关的数据"""
    stems = {Path(name).stem for name in target_filenames}
    print(f"\n🔍 Filtering results for target stems: {stems}")
    
    filtered_stats = []
    filtered_details = []
    filtered_agent_reports = []
    filtered_para_stats = []
    filtered_para_details = []
    
    # 过滤句子级统计和详情
    for i, stat in enumerate(results.get('sent_stats', [])):
        pair = stat['pair']
        if pair[0] in stems:
            filtered_stats.append(stat)
            if i < len(results.get('sent_details', [])):
                filtered_details.append(results['sent_details'][i])
    
    # 过滤 Agent 报告（独立处理，避免重复）
    for report in results.get('agent_reports', []):
        report_pair = report.get('pair', [])
        if len(report_pair) >= 2 and report_pair[0] in stems:
            filtered_agent_reports.append(report)
            print(f"   ✅ Keeping agent report: {report_pair[0]} vs {report_pair[1]}")
    
    # 过滤段落级统计和详情
    for i, pstat in enumerate(results.get('para_stats', [])):
        ppair = pstat['pair']
        if ppair[0] in stems:
            filtered_para_stats.append(pstat)
            if i < len(results.get('para_details', [])):
                filtered_para_details.append(results['para_details'][i])
    
    print(f"   Filtered: {len(filtered_stats)} stats, {len(filtered_details)} details, {len(filtered_agent_reports)} agent reports")
    
    return {
        'sent_stats': filtered_stats,
        'sent_details': filtered_details,
        'para_stats': filtered_para_stats,
        'para_details': filtered_para_details,
        'agent_reports': filtered_agent_reports,
    }


def get_highlight_class(sim, penalty):
    """根据相似度与引用惩罚确定高亮颜色。"""
    if penalty < 0.5:
        return "highlight-citation"
    elif sim >= 0.90:
        return "highlight-high"
    elif sim >= 0.80:
        return "highlight-medium"
    else:
        return "highlight-low"


def read_student_text(temp_dir, student_id):
    """读取指定学生的原始文本内容。"""
    temp_path = Path(temp_dir)
    for file in temp_path.iterdir():
        if file.stem == student_id or file.name.startswith(student_id):
            return file.read_text(encoding='utf-8', errors='ignore')
    return ""


def normalize_pair(pair, target_id):
    """规范化显示顺序：目标文件始终在左侧。"""
    if pair[0] == target_id:
        return pair[0], pair[1]
    else:
        return pair[1], pair[0]


def build_highlighted_text(student_id, text, detail, target_id):
    """构建带相似度高亮的文本HTML片段。"""
    if not text:
        return ""
    
    is_target = (student_id == target_id)
    paragraphs = re.split(r'\n\s*\n', text)
    
    matches = []
    for hit in detail.get('hits', []):
        if is_target:
            if hit['sid_i'] == student_id:
                matches.append({
                    'text': hit['text_i'],
                    'sent_id': hit['sent_id_i'],
                    'sim': hit.get('adjusted_sim', hit['sim']),
                    'penalty': hit.get('citation_penalty', 1.0),
                    'other_text': hit['text_j'],
                    'other_sid': hit['sid_j'],
                })
            elif hit['sid_j'] == student_id:
                matches.append({
                    'text': hit['text_j'],
                    'sent_id': hit['sent_id_j'],
                    'sim': hit.get('adjusted_sim', hit['sim']),
                    'penalty': hit.get('citation_penalty', 1.0),
                    'other_text': hit['text_i'],
                    'other_sid': hit['sid_i'],
                })
        else:
            if hit['sid_j'] == student_id:
                matches.append({
                    'text': hit['text_j'],
                    'sent_id': hit['sent_id_j'],
                    'sim': hit.get('adjusted_sim', hit['sim']),
                    'penalty': hit.get('citation_penalty', 1.0),
                    'other_text': hit['text_i'],
                    'other_sid': hit['sid_i'],
                })
            elif hit['sid_i'] == student_id:
                matches.append({
                    'text': hit['text_i'],
                    'sent_id': hit['sent_id_i'],
                    'sim': hit.get('adjusted_sim', hit['sim']),
                    'penalty': hit.get('citation_penalty', 1.0),
                    'other_text': hit['text_j'],
                    'other_sid': hit['sid_j'],
                })
    
    html_parts = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_html = ""
        sentences = re.split(r'(?<=[。！？.!?;；])', para)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 5:
                continue
            
            match_info = None
            for m in matches:
                if m['text'].strip() in sentence or sentence in m['text'].strip():
                    match_info = m
                    break
            
            if match_info:
                css_class = get_highlight_class(match_info['sim'], match_info['penalty'])
                tooltip = f"Similarity: {match_info['sim']:.1%}"
                if match_info['penalty'] <= 0.60:
                    tooltip += f" (Explicit citation to reference)"
                elif match_info['penalty'] < 1.0:
                    tooltip += f" (Possible citation)"
                para_html += f'<span class="{css_class}" title="{tooltip}">{sentence}</span>'
            else:
                para_html += sentence
        
        html_parts.append(f"<p>{para_html}</p>")
    
    return "".join(html_parts)


def display_comparison_view(detail, temp_dir, target_id, agent_report=None):
    """展示左右对照的文本比较视图（可选Agent分析报告）。"""
    pair = detail['pair']
    
    # Normalize order: target file on left, reference file on right
    left_id, right_id = normalize_pair(pair, target_id)
    
    # Read texts
    text_left = read_student_text(temp_dir, left_id)
    text_right = read_student_text(temp_dir, right_id)
    
    # Statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Similar Sentences", detail['count'])
    with col2:
        st.metric("Avg Similarity", f"{detail['mean_sim']:.1%}")
    with col3:
        st.metric("Text Coverage", f"{detail.get('coverage_min', 0):.1%}")
    with col4:
        score = detail['score']
        if score >= 0.7:
            st.metric("Risk Level", "⚠️ High", delta_color="off")
        elif score >= 0.5:
            st.metric("Risk Level", "⚡ Medium", delta_color="off")
        else:
            st.metric("Risk Level", "✓ Low", delta_color="off")
    with col5:
        st.metric("Source Specificity", f"{detail.get('avg_source_specificity', 0.0):.1%}")
    
    st.divider()
    
    # Legend
    st.markdown("""
    <div style='margin-bottom: 20px;'>
        <div class="legend-item">
            <span class="legend-box" style="background-color: #ff6b6b;"></span>
            <span>High Similarity (≥90%)</span>
        </div>
        <div class="legend-item">
            <span class="legend-box" style="background-color: #ffd93d;"></span>
            <span>Medium Similarity (80-90%)</span>
        </div>
        <div class="legend-item">
            <span class="legend-box" style="background-color: #a8e6cf;"></span>
            <span>Low Similarity (<80%)</span>
        </div>
        <div class="legend-item">
            <span class="legend-box" style="background-color: #d4a5ff; border: 1px dashed #9d4edd;"></span>
            <span>Possible Citation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Side-by-side display
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f'<div class="student-name">🎯 {left_id} (Target)</div>', unsafe_allow_html=True)
        highlighted_left = build_highlighted_text(left_id, text_left, detail, target_id)
        st.markdown(f'<div class="text-container target-file">{highlighted_left}</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown(f'<div class="student-name">📚 {right_id} (Reference)</div>', unsafe_allow_html=True)
        highlighted_right = build_highlighted_text(right_id, text_right, detail, target_id)
        st.markdown(f'<div class="text-container reference-file">{highlighted_right}</div>', unsafe_allow_html=True)
    
    # Display Agent Analysis if available
    if agent_report:
        st.divider()
        st.markdown('<div class="agent-report">', unsafe_allow_html=True)
        st.markdown(agent_report['report'])
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed match list
    with st.expander("📋 View Detailed Match List", expanded=False):
        for i, hit in enumerate(detail.get('hits', [])[:20], 1):
            sim = hit.get('adjusted_sim', hit['sim'])
            penalty = hit.get('citation_penalty', 1.0)
            
            if hit['sid_i'] == left_id:
                left_text = hit['text_i']
                left_sent = hit['sent_id_i']
                right_text = hit['text_j']
                right_sent = hit['sent_id_j']
            else:
                left_text = hit['text_j']
                left_sent = hit['sent_id_j']
                right_text = hit['text_i']
                right_sent = hit['sent_id_i']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{i}. {left_id}** (Sentence {left_sent})")
                st.info(left_text)
            with col2:
                st.markdown(f"**{right_id}** (Sentence {right_sent})")
                st.info(right_text)
            
            label = hit.get('citation_label', '未知')
            if penalty < 1.0:
                st.caption(f"✏️ Similarity: {sim:.1%} | 标记: {label} | 原始: {hit['sim']:.1%}")
            else:
                st.caption(f"Similarity: {sim:.1%}")
            
            if i < len(detail.get('hits', [])[:20]):
                st.divider()

    # Citation label distribution summary
    labels = [hit.get('citation_label', '未知') for hit in detail.get('hits', [])]
    if labels:
        st.divider()
        st.markdown("### 🏷️ Citation Labels Distribution")
        total = len(labels)
        count_clear = sum(1 for l in labels if l == '明确引用')
        count_general = sum(1 for l in labels if l == '一般引用')
        count_none = sum(1 for l in labels if l == '无引用')
        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            st.metric("明确引用", f"{count_clear}/{total}", f"{(count_clear/total):.0%}")
        with col_l2:
            st.metric("一般引用", f"{count_general}/{total}", f"{(count_general/total):.0%}")
        with col_l3:
            st.metric("无引用", f"{count_none}/{total}", f"{(count_none/total):.0%}")


def display_paragraph_view(para_detail, temp_dir, target_id):
    pair = tuple(para_detail['pair'])
    left_id, right_id = normalize_pair(pair, target_id)
    text_left = read_student_text(temp_dir, left_id)
    text_right = read_student_text(temp_dir, right_id)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Similar Paragraphs", para_detail['count'])
    with col2:
        st.metric("Avg Similarity", f"{para_detail['mean_sim']:.1%}")
    with col3:
        st.metric("Max Similarity", f"{para_detail['max_sim']:.1%}")
    with col4:
        st.metric("Coverage", f"{para_detail.get('coverage_min', 0):.1%}")
    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f'<div class="student-name">🎯 {left_id} (Target)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="text-container target-file">{text_left}</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown(f'<div class="student-name">📚 {right_id} (Reference)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="text-container reference-file">{text_right}</div>', unsafe_allow_html=True)
    with st.expander("📋 View Paragraph Matches", expanded=True):
        for i, m in enumerate(para_detail.get('matches', [])[:20], 1):
            colp1, colp2 = st.columns(2)
            with colp1:
                st.markdown(f"**{i}. {left_id}** (Paragraph {m['para_id_i']})")
                st.info(m['text_i'])
            with colp2:
                st.markdown(f"**{right_id}** (Paragraph {m['para_id_j']})")
                st.info(m['text_j'])
            st.caption(f"Similarity: {m['sim']:.1%}")
            if i < len(para_detail.get('matches', [])[:20]):
                st.divider()


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Detection Settings")
    
    device = st.selectbox("Computing Device", ["Auto", "CPU", "GPU"])
    device_map = {"Auto": None, "CPU": "cpu", "GPU": "cuda"}
    
    use_parallel = st.checkbox("CPU Multi-threading", value=True)
    num_workers = st.slider("Thread Count", 1, 8, 2, disabled=not use_parallel)
    
    st.divider()
    
    threshold = st.slider("Sentence Similarity Threshold", 0.50, 0.95, 0.82, 0.01)
    enable_paragraph = st.checkbox("Paragraph Detection", value=True)
    para_threshold = st.slider("Paragraph Threshold", 0.50, 0.90, 0.75, 0.01, disabled=not enable_paragraph)
    
    st.divider()
    
    enable_citation = st.checkbox("Citation Recognition", value=True)
    enable_multilingual = st.checkbox("Cross-lingual Detection", value=False)
    
    st.divider()
    st.subheader("🤖 AI Agent Analysis")
    
    # 检查 API 配置文件
    api_config_exists = Path("api_config.json").exists()
    if not api_config_exists:
        st.error("❌ api_config.json not found! Agent functionality will be disabled.")
        st.info("💡 Create api_config.json in the project root directory with your API credentials.")
    
    enable_agent = st.checkbox(
        "Enable Smart Agent",
        value=False,
        disabled=not api_config_exists,
        help="Use LLM to analyze high-risk pairs and provide detailed reasoning"
    )
    
    # 默认值
    agent_threshold = 0.4  # 降低默认阈值提高触发率
    agent_dual_phase = False
    agent_max_reports = 5
    
    if enable_agent:
        st.success("✅ Agent analysis is ENABLED")
        
        col1, col2 = st.columns(2)
        with col1:
            agent_threshold = st.slider(
                "Agent Trigger Threshold",
                0.3, 1.0, 0.4, 0.05,
                help="Pairs with score ≥ this value will be analyzed by AI. Lower = more pairs analyzed."
            )
        with col2:
            agent_max_reports = st.number_input(
                "Max Reports to Generate",
                min_value=1, max_value=20, value=5,
                help="Maximum number of AI analysis reports (0 = unlimited)"
            )
        
        agent_dual_phase = st.checkbox(
            "Dual-phase analysis (slower but more thorough)",
            value=False,
            help="Enable prosecutor + defender perspectives analysis"
        )
        
        # 显示提示
        st.info(f"💡 With threshold={agent_threshold:.2f}, pairs with similarity score ≥ {agent_threshold:.2f} will trigger AI analysis.")
    else:
        st.info("ℹ️ Agent analysis is disabled. Check the box above to enable.")
    
    st.divider()
    
    if st.button("🗑️ Clear Data"):
        cleanup_temp()
        st.session_state.results = None
        st.session_state.selected_pair = None
        st.session_state.target_file = None
        st.rerun()

# Main Interface
st.title("🔍 Plagiarism Detection System")

tab1, tab2, tab3 = st.tabs(["📁 File Upload", "📊 Comparison Analysis", "🧠 Agent Analysis"])

with tab1:
    st.markdown("### Select Detection Mode")
    
    mode = st.radio(
        "Detection Mode",
        ["Target File Detection", "All Files Comparison"],
        captions=[
            "Upload one target file to compare against multiple reference files",
            "Upload multiple files to detect similarity among all files"
        ],
        horizontal=True
    )
    
    st.divider()
    
    if mode == "Target File Detection":
        st.session_state.detection_mode = "target"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Target Files")
            target_files = st.file_uploader(
                "Upload files to be checked (multiple)",
                type=['txt', 'md'],
                accept_multiple_files=True,
                key='targets'
            )
            if target_files:
                st.success(f"✅ {len(target_files)} target files selected")
                names = [f.name for f in target_files]
                st.session_state.target_files = names
        
        with col2:
            st.markdown("#### 📚 Reference Files")
            reference_files = st.file_uploader(
                "Upload reference files (multiple selection allowed)",
                type=['txt', 'md'],
                accept_multiple_files=True,
                key='references'
            )
            if reference_files:
                st.success(f"✅ {len(reference_files)} reference files selected")
                for rf in reference_files:
                    st.text(f"📄 {rf.name}")
        
        if target_files and reference_files:
            if st.button("🚀 Start Detection", type="primary", use_container_width=True):
                temp_dir = save_uploaded_files(target_files, reference_files)
                
                config_params = {
                    'device': device_map[device],
                    'use_parallel': use_parallel,
                    'num_workers': num_workers,
                    'threshold': threshold,
                    'para_threshold': para_threshold,
                    'enable_paragraph': enable_paragraph,
                    'enable_citation': enable_citation,
                    'enable_multilingual': enable_multilingual,
                    'enable_agent': enable_agent,
                    'agent_threshold': agent_threshold,
                    'agent_max_reports': agent_max_reports if enable_agent else 0,
                    'agent_dual_phase': agent_dual_phase if enable_agent else False,
                    'api_config_path': 'api_config.json',
                    'target_stems': [Path(n).stem for n in st.session_state.target_files],
                    'reference_stems': [Path(rf.name).stem for rf in reference_files],
                }
                
                try:
                    results = run_detection(temp_dir, config_params)
                    filtered = filter_results_by_targets(results, [f.name for f in target_files])
                    st.session_state.results = filtered
                    st.session_state.selected_pair = 0 if filtered['sent_stats'] else None
                    st.success("✅ Detection complete! Switch to 'Comparison Analysis' tab to view results")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Detection failed: {str(e)}")
        elif target_files or reference_files:
            st.warning("⚠️ Please upload both target file and reference files")
    
    else:  # All files comparison
        st.session_state.detection_mode = "all"
        st.session_state.target_file = None
        
        st.markdown("#### 📁 Upload All Files")
        uploaded_files = st.file_uploader(
            "Supports .txt and .md formats, batch upload allowed",
            type=['txt', 'md'],
            accept_multiple_files=True,
            key='all_files'
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files selected")
            cols = st.columns(3)
            for i, f in enumerate(uploaded_files):
                with cols[i % 3]:
                    st.text(f"📄 {f.name}")
        
        if uploaded_files and len(uploaded_files) >= 2:
            if st.button("🚀 Start Detection", type="primary", use_container_width=True):
                temp_dir = save_all_files(uploaded_files)
                
                config_params = {
                    'device': device_map[device],
                    'use_parallel': use_parallel,
                    'num_workers': num_workers,
                    'threshold': threshold,
                    'para_threshold': para_threshold,
                    'enable_paragraph': enable_paragraph,
                    'enable_citation': enable_citation,
                    'enable_multilingual': enable_multilingual,
                    'enable_agent': enable_agent,
                    'agent_threshold': agent_threshold,
                    'agent_max_reports': agent_max_reports if enable_agent else 0,
                    'agent_dual_phase': agent_dual_phase if enable_agent else False,
                    'api_config_path': 'api_config.json',
                }
                
                try:
                    results = run_detection(temp_dir, config_params)
                    st.session_state.results = results
                    if results['sent_stats']:
                        st.session_state.selected_pair = 0
                    st.success("✅ Detection complete! Switch to 'Comparison Analysis' tab to view results")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Detection failed: {str(e)}")
        elif uploaded_files:
            st.warning("⚠️ At least 2 files are required for comparison")

with tab2:
    if st.session_state.results and st.session_state.results['sent_stats']:
        results = st.session_state.results
        stats = results['sent_stats']
        details = results['sent_details']
        agent_reports = results.get('agent_reports', [])
        
        st.markdown("### Detection Results Overview")
        
        target_choice = None
        if st.session_state.detection_mode == "target" and st.session_state.target_files:
            all_targets = sorted({s['pair'][0] for s in stats})
            if all_targets:
                target_choice = st.selectbox("Select target", all_targets, key='target_selector')
        pair_options = []
        indices = []
        for i, stat in enumerate(stats):
            pair = stat['pair']
            if target_choice and pair[0] != target_choice:
                continue
            score = stat['score']
            risk = "🔴 High" if score >= 0.7 else "🟡 Medium" if score >= 0.5 else "🟢 Low"
            spec = stat.get('avg_source_specificity', 0.0)
            pair_options.append(f"{pair[0]} ⟷ {pair[1]} | Score: {score:.3f} | Spec: {spec:.1%} | {risk}")
            indices.append(i)
        
        view_level = "Sentences"
        if results.get('para_stats'):
            view_level = st.radio("View Level", ["Sentences", "Paragraphs"], horizontal=True)
        selected = None
        if view_level == "Sentences":
            selected_idx = st.selectbox(
                "Select a pair to view",
                range(len(pair_options)),
                format_func=lambda x: pair_options[x],
                key='pair_selector'
            )
            selected = indices[selected_idx] if pair_options else None
        else:
            para_options = []
            p_indices = []
            for i, pstat in enumerate(results['para_stats']):
                ppair = pstat['pair']
                if target_choice and ppair[0] != target_choice:
                    continue
                pscore = pstat['score']
                risk = "🔴 High" if pscore >= 0.7 else "🟡 Medium" if pscore >= 0.5 else "🟢 Low"
                para_options.append(f"{ppair[0]} ⟷ {ppair[1]} | Score: {pscore:.3f} | {risk}")
                p_indices.append(i)
            selected_idx = st.selectbox(
                "Select a pair to view",
                range(len(para_options)),
                format_func=lambda x: para_options[x],
                key='para_pair_selector'
            )
            selected = p_indices[selected_idx] if para_options else None
        
        st.divider()
        
        if selected is not None and st.session_state.temp_dir:
            if view_level == "Sentences":
                detail = details[selected]
                if st.session_state.detection_mode == "target" and st.session_state.target_files:
                    target_id = detail['pair'][0]
                else:
                    target_id = detail['pair'][0]
                current_pair = tuple(detail['pair'])
                agent_report = None
                for report in agent_reports:
                    if tuple(report['pair']) == current_pair:
                        agent_report = report
                        break
                display_comparison_view(detail, st.session_state.temp_dir, target_id, agent_report)
            else:
                pdetail = results['para_details'][selected]
                if st.session_state.detection_mode == "target" and st.session_state.target_files:
                    target_id = pdetail['pair'][0]
                else:
                    target_id = pdetail['pair'][0]
                display_paragraph_view(pdetail, st.session_state.temp_dir, target_id)
        
        # Download reports
        st.divider()
        st.markdown("### 📥 Export Reports")
        
        col1, col2, col3, col4 = st.columns(4)

        if st.session_state.temp_dir:
            temp_path = Path(st.session_state.temp_dir)
            
            csv_file = temp_path / "pair_summary.csv"
            json_file = temp_path / "pair_results.json"
            docx_file = temp_path / "plagiarism_report.docx"
            summary_docx = temp_path / "plagiarism_summary_report.docx"
            para_docx = temp_path / "plagiarism_paragraph_report.docx"
            
            with col1:
                st.download_button(
                    "📊 CSV document",
                    csv_file.read_bytes() if csv_file.exists() else b"",
                    "pair_summary.csv",
                    "text/csv",
                    use_container_width=True,
                    disabled=not csv_file.exists(),
                    key="download_csv"
                )
            
            with col2:
                st.download_button(
                    "📄 JSON detail",
                    json_file.read_bytes() if json_file.exists() else b"",
                    "pair_results.json",
                    "application/json",
                    use_container_width=True,
                    disabled=not json_file.exists(),
                    key="download_json"
                )
            
            with col3:
                st.download_button(
                    "📝 Word detailed report",
                    docx_file.read_bytes() if docx_file.exists() else b"",
                    "plagiarism_detailed_report.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    disabled=not docx_file.exists(),
                    key="download_detailed_docx"
                )
            
            with col4:
            # 段落报告下载（如果可用）
                if para_docx.exists() and results.get('para_stats'):
                    st.download_button(
                        "📄 word paragraph report(if applicable)",
                        para_docx.read_bytes(),
                        "plagiarism_paragraph_report.docx",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key="download_para_docx"
                    )
            

        else:
            st.info("👈 Please upload files and start detection first")
        
        st.markdown("""
        ### 💡 Usage Tips
        
        **Target File Detection Mode**:
        - Upload one target file
        - Upload multiple reference files
        - System will check if target file plagiarizes reference files
        - Suitable for checking if student assignments plagiarize online materials
        
        **All Files Comparison Mode**:
        - Upload multiple files
        - System will detect similarity among all files
        - Suitable for batch checking mutual plagiarism among student assignments
        
        **Color Legend**:
        - 🔴 Red: High similarity (≥90%)
        - 🟡 Yellow: Medium similarity (80-90%)
        - 🟢 Green: Low similarity (<80%)
        - 🟣 Purple dashed: Possible citation
        
        **🤖 AI Agent Analysis**:
        - Enable in sidebar settings
        - Uses configured provider to analyze high-risk pairs
        - Provides reasoning, evidence, and possible defenses
        - Requires `api_config.json` with valid API key
        
        **Hover** over highlighted text to view similarity percentage
        """)

with tab3:
    st.markdown("### Agent Reports")
    if st.session_state.results:
        reports = st.session_state.results.get('agent_reports', [])
        stats = st.session_state.results.get('sent_stats', [])
        
        # 调试信息
        print(f"\n🔍 Tab3 Debug Info:")
        print(f"   Results keys: {list(st.session_state.results.keys())}")
        print(f"   Agent reports count: {len(reports)}")
        print(f"   Stats count: {len(stats)}")
        if reports:
            print(f"   First report pair: {reports[0].get('pair', 'N/A')}")
        
        # 显示统计信息
        if stats:
            # 获取当前Agent设置（从sidebar）
            agent_threshold_val = agent_threshold if 'agent_threshold' in locals() else 0.5
            
            total_pairs = len(stats)
            high_risk_pairs = [s for s in stats if s['score'] >= agent_threshold_val]
            
            # 显示风险分数分布
            all_scores = [s['score'] for s in stats]
            max_score = max(all_scores) if all_scores else 0
            min_score = min(all_scores) if all_scores else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pairs", total_pairs)
            with col2:
                st.metric("High Risk Pairs", len(high_risk_pairs))
            with col3:
                st.metric("Agent Reports", len(reports))
            with col4:
                st.metric("Max Score", f"{max_score:.3f}" if max_score > 0 else "N/A")
            
            # 显示所有文档对的分数分布
            st.markdown("#### 📊 Risk Score Distribution:")
            for i, stat in enumerate(sorted(stats, key=lambda x: x['score'], reverse=True)[:10]):  # 显示前10个
                pair = stat['pair']
                score = stat['score']
                status = "✅ Will trigger Agent" if score >= agent_threshold_val else "❌ Below threshold"
                st.text(f"{i+1}. {pair[0]} ⟷ {pair[1]} - Score: {score:.3f} ({status})")
            
            if high_risk_pairs:
                st.success(f"🎯 {len(high_risk_pairs)} pairs meet the Agent threshold (≥{agent_threshold_val:.2f})")
            else:
                st.warning(f"⚠️ No pairs meet the Agent threshold. Try lowering it below {max_score:.3f}")
        
        st.divider()
        
        if reports:
            st.success(f"✅ Found {len(reports)} Agent analysis reports:")
            for i, r in enumerate(reports, 1):
                with st.expander(f"📋 Report {i}: {r['pair'][0]} ⟷ {r['pair'][1]}", expanded=True):
                    st.markdown(r['report'])
        else:
            st.warning("⚠️ No agent reports generated.")
            
            # 详细的诊断信息
            with st.expander("🔧 Diagnostic Information", expanded=False):
                # 检查Agent是否启用
                api_config_exists = Path("api_config.json").exists()
                st.write(f"• Agent enabled in sidebar: {enable_agent if 'enable_agent' in locals() else 'Unknown'}")
                st.write(f"• api_config.json exists: {api_config_exists}")
                
                if stats:
                    max_score = max(s['score'] for s in stats) if stats else 0
                    st.write(f"• Maximum risk score found: {max_score:.3f}")
                    st.write(f"• Current Agent threshold: {agent_threshold_val:.2f}")
                    st.write(f"• High-risk pairs (≥threshold): {len([s for s in stats if s['score'] >= agent_threshold_val])}")
                    
                    # 具体建议
                    if max_score < agent_threshold_val:
                        recommended_threshold = max(0.3, max_score - 0.1)
                        st.warning(f"💡 **Suggestion**: Lower threshold to {recommended_threshold:.2f} to analyze the highest-scoring pair")
                    elif len([s for s in stats if s['score'] >= agent_threshold_val]) == 0:
                        st.warning("💡 **Suggestion**: Try threshold = 0.3 or 0.4 to analyze more pairs")
                
                if api_config_exists:
                    try:
                        with open("api_config.json", 'r') as f:
                            config = json.load(f)
                        st.write(f"• API config loaded successfully")
                        if 'modelscope' in config:
                            st.write(f"• Provider: modelscope")
                            st.write(f"• Model: {config['modelscope'].get('model', 'Unknown')}")
                    except Exception as e:
                        st.write(f"• API config error: {e}")
                
                st.info("💡 **Quick Fix Steps:**\n"
                       "1. **Lower Agent Threshold**: Go to sidebar → Set threshold to 0.4 or 0.5\n"
                       "2. **Check Similarity**: Ensure uploaded files have similar content\n"
                       "3. **Verify API**: Check if api_config.json contains valid credentials\n"
                       "4. **Re-run Detection**: Click 'Start Detection' again after adjusting threshold")
    else:
        st.info("Run detection to generate agent reports.")

st.divider()
st.caption("Plagiarism Detection System v2.1 | Semantic Similarity Analysis with AI-Powered Deep Inspection")
