import io
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ตั้งค่าฟอนต์เริ่มต้นของ Matplotlib เพื่อรองรับภาษาไทย (หากระบบมีฟอนต์ติดตั้งอยู่) และเครื่องหมายคณิตศาสตร์
plt.rcParams['font.sans-serif'] = [
    'Tahoma', 'Leelawadee UI', 'Sarabun', 'TH Sarabun New', 'Ayuthaya', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# ส่วนที่ 1: ฟังก์ชันคำนวณทางคณิตศาสตร์ (พีชคณิตเชิงเส้นและ Eigenvalue/Eigenvector)
# กฎเหล็ก: ห้ามใช้ networkx.pagerank() หรือไลบรารีสำเร็จรูปในการคำนวณเด็ดขาด
# ==============================================================================

def build_link_matrix(edges, n, node_list):
    """
    [M2] สร้าง Link Matrix (L) ขนาด n x n จากรายการลิงก์ (edges)
    
    หลักการทางคณิตศาสตร์และ Eigenvalue:
    - กำหนดให้ คอลัมน์ j คือ "หน้าต้นทาง (Source)" และ แถว i คือ "หน้าปลายทาง (Destination)"
    - สมาชิก L[i, j] แทนสัดส่วนคะแนนโหวตที่หน้า j แบ่งให้หน้า i (Transpose จาก Adjacency Matrix ปกติ)
    - यदिหน้า j มีลิงก์ออกทั้งหมด out_degrees[j] เส้น จะแบ่งคะแนนให้หน้าปลายทางเส้นละ 1 / out_degrees[j]
    - ส่งผลให้ผลรวมของแต่ละคอลัมน์เท่ากับ 1 (Column-stochastic matrix) เมื่อไม่มีหน้าทางตัน
    - สมการสมดุลของคะแนนความสำคัญคือ Lx = 1*x ซึ่งหมายความว่าเวกเตอร์คะแนน x คือ Eigenvector ของ L ที่ λ = 1
    """
    # สร้าง Dictionary แปลงชื่อโหนดเป็นดัชนี (Index 0 ถึง n-1)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # สร้างเมทริกซ์ศูนย์ขนาด n x n สำหรับเก็บค่า Link Matrix L
    L = np.zeros((n, n), dtype=float)
    
    # เวกเตอร์เก็บจำนวนลิงก์ออก (Out-degree) ของแต่ละโหนด
    out_degrees = np.zeros(n, dtype=float)
    
    # กรองลิงก์ซ้ำออกเพื่อให้การแบ่งสัดส่วนคะแนนโหวตของแต่ละหน้าต้นทางคงที่และแม่นยำ
    unique_edges = []
    seen = set()
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            if (u, v) not in seen:
                seen.add((u, v))
                unique_edges.append((u, v))
    
    # นับจำนวนลิงก์ออกทั้งหมดของแต่ละหน้าต้นทาง j
    for u, v in unique_edges:
        j = node_to_idx[u]  # คอลัมน์ j = หน้าต้นทาง
        out_degrees[j] += 1.0
        
    # กำหนดค่าลงใน L[i, j] (แถว i = ปลายทาง, คอลัมน์ j = ต้นทาง)
    for u, v in unique_edges:
        j = node_to_idx[u]  # คอลัมน์ (ต้นทาง)
        i = node_to_idx[v]  # แถว (ปลายทาง)
        if out_degrees[j] > 0:
            L[i, j] = 1.0 / out_degrees[j]
            
    return L, out_degrees


def fix_dangling(L, out_degrees):
    """
    [M2] ตรวจหาและแก้ไขปัญหาหน้าทางตัน (Dangling Nodes) เพื่อสร้างเมทริกซ์ L_hat
    
    หลักการทางคณิตศาสตร์และ Eigenvalue:
    - หน้าทางตันคือหน้าที่ไม่มีลิงก์ออกเลย (out_degrees[j] == 0) ทำให้คอลัมน์ที่ j ของ L เป็น 0 ทั้งคอลัมน์
    - หากรัน Power Iteration บน L ที่มีคอลัมน์ศูนย์ คะแนนรวมจะรั่วไหลและลดลงเรื่อยๆ (λ หลักมีค่าน้อยกว่า 1)
    - วิธีแก้ของ Google คือสมมติให้คนท่องเว็บที่ติดหน้าทางตันสุ่มกระโดดไปทุกหน้าด้วยความน่าจะเป็นเท่ากันคือ 1/n
    - เมื่อแทนคอลัมน์ที่เป็นศูนย์ด้วย 1/n ทุกช่อง จะได้เมทริกซ์ L_hat ที่ผลรวมทุกคอลัมน์เท่ากับ 1 เสมอ
    - ตามสมบัติของ Column-stochastic matrix จะรับประกันทันทีว่า λ = 1 เป็น Eigenvalue ของ L_hat เสมอ
    """
    n = L.shape[0]
    L_hat = L.copy()
    
    # หาตำแหน่งดัชนีของโหนดที่มีจำนวนลิงก์ออกเป็น 0 (หน้าทางตัน)
    dangling_mask = (out_degrees == 0)
    
    # แทนที่คอลัมน์ของหน้าทางตันทุกช่องด้วยค่า 1/n เพื่อรักษาผลรวมคอลัมน์ให้เท่ากับ 1
    if np.any(dangling_mask):
        L_hat[:, dangling_mask] = 1.0 / n
        
    return L_hat, dangling_mask


def google_matrix(L_hat, d, n):
    """
    [M3] คำนวณ Google Matrix (G) จากเมทริกซ์ L_hat และค่า Damping Factor (d)
    
    หลักการทางคณิตศาสตร์และ Eigenvalue:
    - แม้ L_hat จะแก้หน้าทางตันแล้ว แต่กราฟอาจยังมี "เกาะแยก (Disconnected components)" ที่ทำให้ λ = 1 ซ้ำหลายตัว
      (Geometric multiplicity > 1) หรือมี "วงวน (Cycles)" ที่ทำให้มี |λ| = 1 ตัวอื่นบนวงกลมหนึ่งหน่วย จนไม่ลู่เข้า
    - การผสมด้วยพจน์สุ่มกระโดดวาร์ป ((1 - d) / n) * 11^T สร้าง Google Matrix:
      G = d * L_hat + ((1 - d) / n) * np.ones((n, n))
    - เมทริกซ์ G มีสมาชิกทุกตัวเป็นบวกแท้ (G_ij > 0) และผลรวมทุกคอลัมน์เท่ากับ 1
    - ตามทฤษฎีบท Perron-Frobenius:
      1) λ_1 = 1 เป็น Eigenvalue ค่าจริงบวกที่มีขนาดใหญ่ที่สุดเพียงตัวเดียว (Simple eigenvalue)
      2) Eigenvalue ตัวอื่นๆ ของ G จะถูกบีบขนาดลงด้วยอัตราส่วน d คือ λ_i(G) = d * λ_i(L_hat) สำหรับ i >= 2
      3) ดังนั้น |λ_2(G)| <= d < 1 เสมอ รับประกันว่า Power Iteration ลู่เข้าหาคำตอบเดียวแน่นอน
    """
    # สร้างเมทริกซ์ที่ทุกช่องมีค่าเป็น 1 ขนาด n x n (แทน 11^T)
    ones_matrix = np.ones((n, n), dtype=float)
    
    # คำนวณ Google Matrix ตามสมการ G = d * L_hat + ((1 - d) / n) * 11^T
    G = d * L_hat + ((1.0 - d) / n) * ones_matrix
    return G


def power_step(G, x):
    """
    [M5] คำนวณ Power Iteration ก้าวต่อไป 1 รอบ: x_{k+1} = G @ x_k
    
    หลักการทางคณิตศาสตร์และ Eigenvalue:
    - จากการแยกตัวประกอบเมทริกซ์ G = P D P^{-1} เมื่อกระจายเวกเตอร์เริ่มต้น x_0 ในฐานของ Eigenvector:
      x_0 = c_1 * v_1 + c_2 * v_2 + ... + c_n * v_n
    - เมื่อคูณ G ซ้ำ k รอบ:
      x_k = G^k x_0 = P D^k P^{-1} x_0 = c_1 (1)^k v_1 + c_2 (λ_2)^k v_2 + ... + c_n (λ_n)^k v_n
    - เนื่องจาก |λ_i| <= |λ_2| <= d < 1 สำหรับทุก i >= 2 พจน์ชั่วคราว (Transient terms) จึงหดตัวเข้าใกล้ 0
      ด้วยอัตราเร็วที่ถูกควบคุมโดย |λ_2|^k เหลือรอดเพียงส่วนของ v_1 (ซึ่งสมนัยกับ λ_1 = 1)
    """
    # คูณเมทริกซ์ G กับเวกเตอร์สถานะปัจจุบัน x
    x_next = G @ x
    
    # ปรับให้ผลรวมเป็น 1 เสมอเพื่อป้องกันความคลาดเคลื่อนของการปัดเศษทศนิยม (Floating-point drift)
    x_next = x_next / np.sum(x_next)
    return x_next


def eigen_pagerank(G):
    """
    [M4, M7] คำนวณหาค่า PageRank โดยตรงด้วย Eigen-decomposition (Gx = λx ที่ λ = 1)
    
    หลักการทางคณิตศาสตร์และ Eigenvalue:
    - แก้สมการ Gx = λx ด้วย np.linalg.eig(G) เพื่อดึง Eigenvalues ทั้งหมดและ Eigenvectors
    - ค้นหาดัชนีของ Eigenvalue ที่มีค่าใกล้เคียง λ = 1.0 มากที่สุด (Dominant eigenvalue λ_1 = 1)
    - ดึงคอลัมน์ Eigenvector ที่สมนัยกันออกมา ตัดเอาเฉพาะส่วนจริง (.real)
    - นำไปหารด้วยผลรวมของตัวเอง (pi / pi.sum()) เพื่อปรับสเกลให้ผลรวมความน่าจะเป็นเท่ากับ 1 และเป็นบวกเสมอ
    - คำนวณหา |λ_2| (ขนาดของ Eigenvalue ที่ใหญ่อันดับสอง) เพื่อใช้วิเคราะห์อัตราการลู่เข้า
    """
    # คำนวณหา Eigenvalues (lam) และเมทริกซ์ของ Eigenvectors (P)
    lam, P = np.linalg.eig(G)
    
    # ค้นหาตำแหน่ง index ของ Eigenvalue ที่อยู่ใกล้ 1.0 ที่สุดบนระนาบเชิงซ้อน
    idx_1 = np.argmin(np.abs(lam - 1.0))
    
    # ดึง Eigenvector คอลัมน์ที่สมนัยกับ λ = 1 และเอาเฉพาะส่วนจริง (.real)
    v1_real = P[:, idx_1].real
    
    # หารด้วยผลรวมของตัวเอง เพื่อให้คะแนนรวมทุกหน้าเท่ากับ 1 เสมอ (และกลับเครื่องหมายอัตโนมัติหากติดลบทั้งเวกเตอร์)
    pi = v1_real / np.sum(v1_real)
    
    # บังคับค่าศูนย์หากมีค่าติดลบขนาดจิ๋วระดับเครื่องหมายทศนิยม (เช่น -1e-17) แล้วปรับผลรวมอีกครั้ง
    pi = np.maximum(pi, 0.0)
    pi = pi / np.sum(pi)
    
    # เรียงลำดับขนาด |λ| จากน้อยไปมาก เพื่อดึงค่าใหญ่อันดับสอง |λ_2|
    abs_lam_sorted = np.sort(np.abs(lam))
    lam2_abs = float(abs_lam_sorted[-2]) if len(abs_lam_sorted) >= 2 else 0.0
    
    return pi, lam, lam2_abs


# ==============================================================================
# ส่วนที่ 2: ข้อมูลกราฟตัวอย่าง (Sample Graphs สำหรับ M1 และ M8)
# ==============================================================================

SAMPLE_GRAPHS = {
    "หน้าทางตัน": (
        "A,B\nA,C\nB,C\nC,A\nC,D\nD,E\nE,C\nE,F",
        "กราฟ 6 หน้า ที่มี 'F' เป็นหน้าทางตัน (Dangling Node) ไม่มีลิงก์ออก ระบบจะปรับคอลัมน์ F เป็น 1/n อัตโนมัติ"
    ),
    "สองเกาะ": (
        "A,B\nB,C\nC,A\nD,E\nE,F\nF,D",
        "กราฟ 2 เกาะแยกจากกันโดยสิ้นเชิง (A-B-C และ D-E-F) ถ้าไม่มี Damping factor ค่า λ=1 จะซ้ำ 2 ตัว แต่เมื่อใช้ G จะได้คำตอบเดียว"
    ),
    "วนเป็นวงกลม": (
        "A,B\nB,C\nC,D\nD,E\nE,A",
        "กราฟวงกลม 5 หน้า (A→B→C→D→E→A) มี Eigenvalue เชิงซ้อนกระจายตัวเป็นรูปห้าเหลี่ยมบนวงกลมรัศมี d ทำให้เกิดการแกว่งก่อนลู่เข้า"
    ),
    "ข้อมูลของกลุ่ม": (
        "Calc1,Calc2\nCalc1,LinAlg\nCalc2,DiffEq\nCalc2,ProbStat\n"
        "LinAlg,ML\nLinAlg,Optimiz\nLinAlg,DataSci\nLinAlg,DeepLearn\n"
        "Prog1,DataStruct\nProg1,DataSci\nDataStruct,Algo\nDataStruct,ML\n"
        "ProbStat,ML\nProbStat,DataSci\nAlgo,Optimiz\nAlgo,DeepLearn\n"
        "ML,DeepLearn\nDataSci,ML\nOptimiz,ML\nDiffEq,Optimiz",
        "ข้อมูลจริงของกลุ่ม (12 รายวิชา): โครงข่ายการอ้างอิงความรู้ระหว่างวิชาในสายวิทยาการข้อมูลและ AI (วิชาพื้นฐานชี้ไปยังวิชาขั้นสูงที่ต้องใช้ความรู้นั้นต่อยอด)"
    )
}


def parse_edges_from_text(raw_text):
    """แปลงข้อความบรรทัดละลิงก์ (from,to) ให้เป็นรายการ edges และรายชื่อโหนดที่เรียงลำดับแล้ว"""
    edges = []
    nodes_set = set()
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            # ข้าม header หากผู้ใช้เผลอใส่คำว่า from,to
            if parts[0].lower() == "from" and parts[1].lower() == "to":
                continue
            u, v = parts[0], parts[1]
            edges.append((u, v))
            nodes_set.add(u)
            nodes_set.add(v)
    node_list = sorted(list(nodes_set))
    return edges, node_list


def reset_power_iteration(n, G, pi):
    """รีเซ็ตสถานะของ Power Iteration ใน st.session_state ให้กลับไปที่รอบ k = 0"""
    x0 = np.ones(n, dtype=float) / n
    err0 = float(np.linalg.norm(x0 - pi, ord=2))
    st.session_state.k = 0
    st.session_state.x_k = x0
    st.session_state.error_history = [err0]
    st.session_state.x_history = [x0.copy()]


# ==============================================================================
# ส่วนที่ 3: การจัดวางหน้าจอ Streamlit UI (Dashboard Layout)
# ==============================================================================

st.set_page_config(
    page_title="PageRank Lab · SCMA 251",
    layout="wide"
)

# กำหนดค่าเริ่มต้นใน st.session_state
if "edges_text" not in st.session_state:
    st.session_state.edges_text = SAMPLE_GRAPHS["หน้าทางตัน"][0]
if "sample_desc" not in st.session_state:
    st.session_state.sample_desc = SAMPLE_GRAPHS["หน้าทางตัน"][1]
if "state_signature" not in st.session_state:
    st.session_state.state_signature = None

# ------------------------------------------------------------------------------
# แถบด้านซ้าย (Sidebar): ควบคุม Input [M1], Damping Factor [M3], และ Power Iteration [M5]
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🕸️ กราฟลิงก์ (`from,to`) [M1]")
    
    # ปุ่มโหลดกราฟตัวอย่าง 4 รูปแบบตามที่กำหนด
    st.caption("**เลือกโหลดกราฟตัวอย่าง:**")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("หน้าทางตัน", use_container_width=True):
            st.session_state.edges_text = SAMPLE_GRAPHS["หน้าทางตัน"][0]
            st.session_state.sample_desc = SAMPLE_GRAPHS["หน้าทางตัน"][1]
            st.rerun()
        if st.button("วนเป็นวงกลม", use_container_width=True):
            st.session_state.edges_text = SAMPLE_GRAPHS["วนเป็นวงกลม"][0]
            st.session_state.sample_desc = SAMPLE_GRAPHS["วนเป็นวงกลม"][1]
            st.rerun()
    with btn_col2:
        if st.button("สองเกาะ", use_container_width=True):
            st.session_state.edges_text = SAMPLE_GRAPHS["สองเกาะ"][0]
            st.session_state.sample_desc = SAMPLE_GRAPHS["สองเกาะ"][1]
            st.rerun()
        if st.button("ข้อมูลของกลุ่ม", use_container_width=True):
            st.session_state.edges_text = SAMPLE_GRAPHS["ข้อมูลของกลุ่ม"][0]
            st.session_state.sample_desc = SAMPLE_GRAPHS["ข้อมูลของกลุ่ม"][1]
            st.rerun()

    # File Uploader สำหรับอัปโหลดไฟล์ CSV
    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ CSV (คอลัมน์ `from,to` รองรับ 3–20+ หน้า)",
        type=["csv"]
    )
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            df_csv = pd.read_csv(io.StringIO(content))
            if {"from", "to"}.issubset(set(df_csv.columns.str.lower())):
                # ดึงตามชื่อคอลัมน์ไม่ว่าจะพิมพ์เล็กหรือใหญ่
                col_map = {c.lower(): c for c in df_csv.columns}
                u_col, v_col = col_map["from"], col_map["to"]
                lines = [f"{row[u_col]},{row[v_col]}" for _, row in df_csv.iterrows()]
            else:
                # กรณีไม่มี header ใช้ 2 คอลัมน์แรก
                df_csv_no_hdr = pd.read_csv(io.StringIO(content), header=None)
                lines = [f"{row[0]},{row[1]}" for _, row in df_csv_no_hdr.iterrows()]
            csv_text = "\n".join(lines)
            if st.button("📥 นำเข้าข้อมูลจากไฟล์ CSV นี้", use_container_width=True):
                st.session_state.edges_text = csv_text
                st.session_state.sample_desc = f"อัปโหลดจากไฟล์ CSV: {uploaded_file.name}"
                st.rerun()
        except Exception as e:
            st.error(f"ไม่สามารถอ่านไฟล์ CSV ได้: {e}")

    # ช่อง Text Area สำหรับกรอกหรือแก้ไขลิงก์โดยตรง
    edges_input = st.text_area(
        "รายการลิงก์ (บรรทัดละ 1 ลิงก์ รูปแบบ `ต้นทาง,ปลายทาง`):",
        value=st.session_state.edges_text,
        height=175
    )
    if edges_input != st.session_state.edges_text:
        st.session_state.edges_text = edges_input
        st.session_state.sample_desc = "กำหนดข้อมูลกราฟเองผ่านช่องข้อความ"

    # แปลงข้อมูลลิงก์เป็นโหนดและเมทริกซ์
    edges, node_list = parse_edges_from_text(st.session_state.edges_text)
    n = len(node_list)

    if n < 2:
        st.error("กรุณากรอกข้อมูลลิงก์อย่างน้อย 2–3 หน้าขึ้นไป (รูปแบบ `A,B`)")
        st.stop()

    # คำนวณ Link Matrix L และตรวจหาหน้าทางตัน [M2]
    L, out_degrees = build_link_matrix(edges, n, node_list)
    L_hat, dangling_mask = fix_dangling(L, out_degrees)
    dangling_nodes = [node_list[i] for i in range(n) if dangling_mask[i]]

    # แจ้งเตือนหน้าทางตันบนหน้าจอทันทีหากตรวจพบ [M2]
    if dangling_nodes:
        st.warning(
            f"⚠️ **พบหน้าทางตัน (Dangling Node):** `{', '.join(dangling_nodes)}` "
            f"(ไม่มีลิงก์ออก คอลัมน์เป็น 0) → ระบบได้แทนคอลัมน์ด้วย $1/{n}$ เพื่อปรับเป็น $\\hat{{L}}$ แล้ว [M2]"
        )
    else:
        st.success("✅ ไม่พบหน้าทางตัน ทุกหน้ามีลิงก์ออก ($L = \\hat{L}$) [M2]")

    st.markdown("---")
    # [M3] Slider ปรับค่า Damping Factor d
    st.markdown("### 🎚️ Damping Factor $d$ [M3]")
    d = st.slider(
        "ปรับโอกาสกดตามลิงก์ปกติ ($d$):",
        min_value=0.50,
        max_value=0.99,
        value=0.85,
        step=0.01,
        help="โอกาส d ที่คนท่องเว็บจะคลิกลิงก์ในหน้าปัจจุบัน และโอกาส (1-d) ที่จะสุ่มกระโดดวาร์ปไปหน้าใดก็ได้"
    )

    # สร้าง Google Matrix G และหา Eigenvector ทันทีเมื่อเลื่อน Slider [M3, M4]
    G = google_matrix(L_hat, d, n)
    pi, eigvals, lam2_abs = eigen_pagerank(G)

    # ตรวจสอบว่ากราฟหรือค่า d มีการเปลี่ยนแปลงหรือไม่ เพื่อรีเซ็ตหรือซิงค์ st.session_state
    current_signature = (tuple(edges), tuple(node_list), round(d, 4))
    if st.session_state.state_signature != current_signature:
        reset_power_iteration(n, G, pi)
        st.session_state.state_signature = current_signature

    st.markdown("---")
    # [M5] แผงควบคุม Power Iteration
    st.markdown("### ⚡ Power Iteration [M5]")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("▶️ ก้าวไป 1 รอบ", use_container_width=True):
            x_next = power_step(G, st.session_state.x_k)
            st.session_state.x_k = x_next
            st.session_state.k += 1
            err = float(np.linalg.norm(x_next - pi, ord=2))
            st.session_state.error_history.append(err)
            st.session_state.x_history.append(x_next.copy())

    with p_col2:
        if st.button("⏩ รันจนลู่เข้า", use_container_width=True):
            # วนลูปคูณ G @ x ซ้ำจนกว่า error < 1e-6 หรือครบ 1000 รอบ
            while st.session_state.k < 1000:
                curr_err = float(np.linalg.norm(st.session_state.x_k - pi, ord=2))
                if curr_err < 1e-6 and st.session_state.k > 0:
                    break
                x_next = power_step(G, st.session_state.x_k)
                st.session_state.x_k = x_next
                st.session_state.k += 1
                new_err = float(np.linalg.norm(x_next - pi, ord=2))
                st.session_state.error_history.append(new_err)
                st.session_state.x_history.append(x_next.copy())
                if new_err < 1e-6:
                    break

    if st.button("🔄 เริ่มใหม่ ($\\vec{x}_0 = 1/n$ ทุกหน้า)", use_container_width=True):
        reset_power_iteration(n, G, pi)
        st.rerun()

    current_err = float(np.linalg.norm(st.session_state.x_k - pi, ord=2))
    st.info(
        f"**รอบปัจจุบัน:** $k = {st.session_state.k}$ (tol $10^{{-6}}$)\n\n"
        f"**Error ปัจจุบัน $\\|\\vec{{x}}_k - \\vec{{\\pi}}\\|_2$:** `{current_err:.3e}`"
    )


# ------------------------------------------------------------------------------
# พื้นที่หลัก (Main Dashboard): แสดงกราฟ, ตารางเปรียบเทียบ, Eigenvalues และ Error Plot
# ------------------------------------------------------------------------------
st.title("PageRank Lab")
st.caption(
    f" "
    f""
)

# แถวบน: (1) กราฟโครงข่าย [M1, M7] และ (2) ตารางเปรียบเทียบอันดับ [M7]
top_col1, top_col2 = st.columns([1.15, 1.0], gap="medium")

with top_col1:
    with st.container(border=True):
        st.subheader("(1) โครงข่ายกราฟลิงก์ (ขนาดโหนด = คะแนน PageRank) [M1, M7]")
        
        # สร้างกราฟแบบมีทิศทางด้วย networkx (ใช้สำหรับการวาดภาพเท่านั้นตามกฎเหล็ก)
        DG = nx.DiGraph()
        DG.add_nodes_from(node_list)
        DG.add_edges_from(edges)

        fig_g, ax_g = plt.subplots(figsize=(6.5, 4.8), dpi=110)
        
        # จัดวางตำแหน่งโหนดให้สวยงามตามลักษณะกราฟ
        if n <= 8:
            pos = nx.kamada_kawai_layout(DG)
        else:
            pos = nx.spring_layout(DG, k=1.3 / np.sqrt(n), seed=42, iterations=80)

        # ใช้คะแนนปัจจุบันจาก Power Iteration (x_k) มาปรับขนาดโหนดให้เห็นการเปลี่ยนแปลงแต่ละรอบ
        scores_for_size = st.session_state.x_k
        max_score = np.max(scores_for_size) if np.max(scores_for_size) > 0 else 1.0
        node_sizes = [600 + 2600 * (s / max_score) for s in scores_for_size]

        # แยกสีและเส้นขอบ: หน้าทางตันใช้สีชมพูอ่อนขอบประ โหนดที่มีคะแนนสูงสุดใช้สีส้มอ่อน โหนดทั่วไปสีฟ้าอ่อน
        top_node_idx = int(np.argmax(scores_for_size))
        node_colors = []
        edge_colors = []
        for idx in range(n):
            if dangling_mask[idx]:
                node_colors.append("#FADBD8")
                edge_colors.append("#C0392B")
            elif idx == top_node_idx:
                node_colors.append("#FDEBD0")
                edge_colors.append("#D35400")
            else:
                node_colors.append("#D6EAF8")
                edge_colors.append("#2874A6")

        # วาดโหนดปกติและโหนดหน้าทางตันแยกกันเพื่อให้หน้าทางตันแสดงขอบเส้นประชัดเจน
        normal_indices = [i for i in range(n) if not dangling_mask[i]]
        dangling_indices = [i for i in range(n) if dangling_mask[i]]

        if normal_indices:
            nx.draw_networkx_nodes(
                DG, pos,
                nodelist=[node_list[i] for i in normal_indices],
                node_size=[node_sizes[i] for i in normal_indices],
                node_color=[node_colors[i] for i in normal_indices],
                edgecolors=[edge_colors[i] for i in normal_indices],
                linewidths=2.0,
                ax=ax_g
            )
        if dangling_indices:
            nx.draw_networkx_nodes(
                DG, pos,
                nodelist=[node_list[i] for i in dangling_indices],
                node_size=[node_sizes[i] for i in dangling_indices],
                node_color=[node_colors[i] for i in dangling_indices],
                edgecolors=[edge_colors[i] for i in dangling_indices],
                linewidths=2.0,
                ax=ax_g
            )
            # ตั้งค่าเส้นประให้กับหน้าทางตัน
            ax_g.collections[-1].set_linestyle("--")

        # วาดป้ายชื่อโหนดพร้อมคะแนน
        labels = {node_list[i]: f"{node_list[i]}\n({scores_for_size[i]:.3f})" for i in range(n)}
        nx.draw_networkx_labels(
            DG, pos,
            labels=labels,
            font_size=9 if n <= 10 else 7.5,
            font_weight="bold",
            ax=ax_g
        )

        # วาดเส้นเชื่อมพร้อมหัวลูกศรโค้งเล็กน้อยเพื่อให้เห็นลิงก์สวนทางกัน (เช่น A->C และ C->A) ได้ชัดเจน
        nx.draw_networkx_edges(
            DG, pos,
            edgelist=list(DG.edges()),
            edge_color="#34495E",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            width=1.6,
            connectionstyle="arc3,rad=0.12",
            node_size=node_sizes,
            ax=ax_g
        )

        ax_g.set_axis_off()
        ax_g.margins(0.15)
        fig_g.tight_layout()
        st.pyplot(fig_g)
        plt.close(fig_g)
        st.caption(
            "💡 **คำอธิบายสีโหนด:** 🟠 สีส้ม = อันดับ 1 ปัจจุบัน · 🔵 สีฟ้า = หน้าทั่วไป · "
            "🔴 ขอบเส้นประสีแดง = หน้าทางตัน (Dangling Node)"
        )

with top_col2:
    with st.container(border=True):
        st.subheader("(2) ตารางอันดับ: เปรียบเทียบ 2 วิธีต้องตรงกัน [M5, M7]")
        
        # สร้าง DataFrame เปรียบเทียบค่าจาก Power Iteration (x_k) และ Eigenvector (pi)
        df_rank = pd.DataFrame({
            "หน้า (Node)": node_list,
            f"Power Iter. (x_k, k={st.session_state.k})": st.session_state.x_k,
            "Eigenvector (λ = 1)": pi,
            "ผลต่าง |x_k - π|": np.abs(st.session_state.x_k - pi),
            "ลิงก์ออก (Out)": out_degrees.astype(int)
        })
        
        # เรียงลำดับตามคะแนน Eigenvector จากมากไปน้อย
        df_rank = df_rank.sort_values(by="Eigenvector (λ = 1)", ascending=False).reset_index(drop=True)
        df_rank.index = df_rank.index + 1
        df_rank.index.name = "อันดับ"

        st.dataframe(
            df_rank.style.format({
                f"Power Iter. (x_k, k={st.session_state.k})": "{:.6f}",
                "Eigenvector (λ = 1)": "{:.6f}",
                "ผลต่าง |x_k - π|": "{:.2e}"
            }),
            use_container_width=True,
            height=290
        )

        max_diff = float(np.max(np.abs(st.session_state.x_k - pi)))
        if max_diff < 1e-5:
            st.success(f"✓ ทั้ง 2 วิธีให้ผลลัพธ์ตรงกัน! ผลต่างมากสุดเพียง `{max_diff:.2e}` (ลู่เข้าสู่ Eigenvector ของ $\\lambda=1$ แล้ว)")
        else:
            st.warning(
                f"⏳ ปัจจุบันอยู่ที่รอบ $k = {st.session_state.k}$ (ผลต่างมากสุด `{max_diff:.4e}`) — "
                f"กด **'ก้าวไป 1 รอบ'** หรือ **'รันจนลู่เข้า'** ที่แถบซ้ายเพื่อดู $\\vec{{x}}_k$ ลู่เข้าหา $\\vec{{\\pi}}$"
            )

        # แสดงเวกเตอร์ x_k ปัจจุบันในรูปแบบพร้อมคัดลอกหรือตรวจสอบ [M5]
        with st.expander(f"🔎 ดูเวกเตอร์ปัจจุบัน x_k (รอบที่ k = {st.session_state.k}) และค่าผลรวม"):
            vec_df = pd.DataFrame([st.session_state.x_k], columns=node_list, index=[f"x_{st.session_state.k}"])
            st.dataframe(vec_df.style.format("{:.6f}"), use_container_width=True)
            st.caption(f"ผลรวมคะแนนทุกหน้า $\\sum x_k = {np.sum(st.session_state.x_k):.6f}$ (คงที่เท่ากับ 1 เสมอ)")


# แถวล่าง: (3) Eigenvalues บนระนาบเชิงซ้อน [M4] และ (4) กราฟ Error เทียบกับ |λ_2|^k [M6]
bot_col1, bot_col2 = st.columns([1.0, 1.15], gap="medium")

with bot_col1:
    with st.container(border=True):
        st.subheader("(3) Eigenvalues ของ $G$ บนระนาบเชิงซ้อน [M4]")
        
        fig_eig, ax_eig = plt.subplots(figsize=(5.5, 4.5), dpi=110)
        theta = np.linspace(0, 2 * np.pi, 300)
        
        # วาดวงกลมรัศมี 1 (|λ| = 1) และวงกลมรัศมี d (|λ| = d)
        ax_eig.plot(np.cos(theta), np.sin(theta), linestyle="--", color="#7F8C8D", linewidth=1.3, label=r"$|\lambda| = 1$")
        ax_eig.plot(d * np.cos(theta), d * np.sin(theta), linestyle="--", color="#8E44AD", linewidth=1.5, label=rf"$|\lambda| = d = {d:.2f}$")

        # วาดแกน Re และ Im ผ่านจุดกำเนิด
        ax_eig.axhline(0, color="#BDC3C7", linewidth=1.0, zorder=1)
        ax_eig.axvline(0, color="#BDC3C7", linewidth=1.0, zorder=1)

        # แยกจุด λ_1 = 1 กับ Eigenvalue ตัวอื่นๆ เพื่อพล็อตคนละสีให้ชัดเจน
        idx_1 = int(np.argmin(np.abs(eigvals - 1.0)))
        other_mask = np.ones(len(eigvals), dtype=bool)
        other_mask[idx_1] = False

        ax_eig.scatter(
            eigvals[other_mask].real,
            eigvals[other_mask].imag,
            color="#1F618D",
            s=55,
            zorder=4,
            label=rf"Other $\lambda_i$ ($|\lambda_2| = {lam2_abs:.4f}$)"
        )
        ax_eig.scatter(
            [eigvals[idx_1].real],
            [eigvals[idx_1].imag],
            color="#C0392B",
            s=90,
            zorder=5,
            label=r"$\lambda_1 = 1$ (PageRank)"
        )

        ax_eig.set_xlim(-1.18, 1.18)
        ax_eig.set_ylim(-1.18, 1.18)
        ax_eig.set_aspect("equal", adjustable="box")
        ax_eig.set_xlabel("Real (Re)")
        ax_eig.set_ylabel("Imaginary (Im)")
        ax_eig.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        ax_eig.grid(True, linestyle=":", alpha=0.4)
        fig_eig.tight_layout()
        st.pyplot(fig_eig)
        plt.close(fig_eig)

        # แสดงค่า |λ_2| บนหน้าจอชัดเจนตามข้อกำหนด [M4]
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Dominant λ₁", "1.0000")
        m_c2.metric("ขนาด |λ₂|", f"{lam2_abs:.4f}")
        m_c3.metric("ขอบเขตทฤษฎี d", f"{d:.2f}", delta=f"|λ₂| ≤ d ({lam2_abs <= d + 1e-9})")

with bot_col2:
    with st.container(border=True):
        st.subheader("(4) กราฟ Error $\\|\\vec{x}_k - \\vec{\\pi}\\|_2$ เทียบกับ $|\\lambda_2|^k$ [M6]")
        
        fig_err, ax_err = plt.subplots(figsize=(6.2, 4.5), dpi=110)
        
        k_vals = np.arange(len(st.session_state.error_history))
        err_vals = np.array(st.session_state.error_history, dtype=float)
        # ป้องกัน log(0) ด้วยการกำหนดค่าต่ำสุดที่ 1e-16
        err_vals_clipped = np.maximum(err_vals, 1e-16)

        # พล็อตเส้น Error จริงจากการคูณ Power Iteration
        ax_err.semilogy(
            k_vals,
            err_vals_clipped,
            marker="o",
            markersize=4 if len(k_vals) < 40 else 2,
            linewidth=2.0,
            color="#1F618D",
            label=r"Actual Error $\|\vec{x}_k - \vec{\pi}\|_2$"
        )

        # พล็อตเส้นแนวโน้มทางทฤษฎี |λ_2|^k (ทั้งแบบปรับสเกลด้วย Error เริ่มต้น e_0 * |λ_2|^k และ |λ_2|^k บริสุทธิ์)
        max_k_plot = max(int(k_vals[-1]), 15)
        k_theo = np.arange(0, max_k_plot + 1)
        if lam2_abs > 1e-12:
            e0 = err_vals_clipped[0]
            theo_scaled = np.maximum(e0 * (lam2_abs ** k_theo), 1e-16)
            theo_raw = np.maximum(lam2_abs ** k_theo, 1e-16)
            ax_err.semilogy(
                k_theo,
                theo_scaled,
                linestyle="--",
                linewidth=1.8,
                color="#C0392B",
                label=rf"Theoretical Decay $\|\vec{{x}}_0-\vec{{\pi}}\|\cdot |\lambda_2|^k$ ($|\lambda_2|={lam2_abs:.3f}$)"
            )
            ax_err.semilogy(
                k_theo,
                theo_raw,
                linestyle=":",
                linewidth=1.4,
                color="#E67E22",
                label=rf"Pure $|\lambda_2|^k = ({lam2_abs:.3f})^k$"
            )

        ax_err.axhline(1e-6, color="#27AE60", linestyle="-.", linewidth=1.1, label="Tolerance ($10^{-6}$)")
        ax_err.set_xlabel("Iteration ($k$)")
        ax_err.set_ylabel(r"Error $\|\vec{x}_k - \vec{\pi}\|_2$ (Log Scale)")
        ax_err.set_xlim(0, max_k_plot)
        ax_err.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_err.legend(loc="upper right", fontsize=8.5)
        fig_err.tight_layout()
        st.pyplot(fig_err)
        plt.close(fig_err)

        st.caption(
            f"📉 **ความหมายทางพีชคณิตเชิงเส้น:** จาก $\\vec{{x}}_k = G^k \\vec{{x}}_0 = \\vec{{\\pi}} + c_2 \\lambda_2^k \\vec{{v}}_2 + \\dots$ "
            f"ความชันบนกราฟ Log-scale จึงขนานกับเส้น $|\\lambda_2|^k = ({lam2_abs:.4f})^k$ พอดี"
        )


# ------------------------------------------------------------------------------
# ส่วนแสดงเมทริกซ์ L, L_hat และ Google Matrix G อย่างละเอียด [M2, M3]
# ------------------------------------------------------------------------------
st.markdown("### 🧮 ตรวจสอบเมทริกซ์การคำนวณ ($L$, $\\hat{L}$, และ $G$) [M2, M3]")
st.caption("คอลัมน์ = หน้าต้นทาง ($j$) · แถว = หน้าปลายทาง ($i$) · ผลรวมของแต่ละคอลัมน์ใน $\\hat{L}$ และ $G$ มีค่าเท่ากับ 1.0000 เสมอ")

tab_L, tab_Lhat, tab_G, tab_eig = st.tabs([
    "1️⃣ Link Matrix (L) [M2]",
    "2️⃣ Modified Link Matrix (L̂) [M2]",
    f"3️⃣ Google Matrix (G, d={d:.2f}) [M3]",
    "4️⃣ รายการ Eigenvalues ทั้งหมดของ G [M4]"
])

with tab_L:
    st.markdown(
        "**Link Matrix ($L$):** สมาชิก $L_{i,j}$ คือสัดส่วนคะแนนที่หน้าต้นทาง $j$ (คอลัมน์) แบ่งให้หน้าปลายทาง $i$ (แถว) "
        "หากหน้า $j$ เป็นหน้าทางตัน คอลัมน์ที่ $j$ จะเป็น 0 ทั้งหมด"
    )
    df_L = pd.DataFrame(L, index=node_list, columns=node_list)
    df_L.loc["ผลรวมคอลัมน์ (Col Sum)"] = df_L.sum(axis=0)
    st.dataframe(df_L.style.format("{:.4f}"), use_container_width=True)

with tab_Lhat:
    st.markdown(
        "**Modified Link Matrix ($\\hat{L}$):** แก้ไขหน้าทางตันโดยแทนคอลัมน์ที่เป็นศูนย์ทั้งหมดด้วย $1/n = "
        f"1/{n} = {1.0/n:.4f}$ เพื่อให้ผลรวมทุกคอลัมน์เท่ากับ 1"
    )
    df_Lhat = pd.DataFrame(L_hat, index=node_list, columns=node_list)
    df_Lhat.loc["ผลรวมคอลัมน์ (Col Sum)"] = df_Lhat.sum(axis=0)
    st.dataframe(df_Lhat.style.format("{:.4f}"), use_container_width=True)

with tab_G:
    teleport_val = (1.0 - d) / n
    st.markdown(
        f"**Google Matrix ($G = d\\hat{{L}} + \\frac{{1-d}}{{n}}\\mathbf{{1}}\\mathbf{{1}}^T$):** "
        f"เมื่อ $d = {d:.2f}$ และพจน์สุ่มกระโดดวาร์ป $\\frac{{1-d}}{{n}} = \\frac{{{1-d:.2f}}}{{{n}}} = {teleport_val:.4f}$"
    )
    df_G = pd.DataFrame(G, index=node_list, columns=node_list)
    df_G.loc["ผลรวมคอลัมน์ (Col Sum)"] = df_G.sum(axis=0)
    st.dataframe(df_G.style.format("{:.4f}"), use_container_width=True)

with tab_eig:
    # เรียงลำดับ Eigenvalues ตามขนาด |λ| จากมากไปน้อยเพื่อแสดงในตาราง
    order_idx = np.argsort(np.abs(eigvals))[::-1]
    eig_sorted = eigvals[order_idx]
    df_eig = pd.DataFrame({
        "ลำดับ": [f"λ_{i+1}" for i in range(n)],
        "ส่วนจริง Re(λ)": eig_sorted.real,
        "ส่วนจินตภาพ Im(λ)": eig_sorted.imag,
        "ขนาด |λ|": np.abs(eig_sorted)
    })
    st.dataframe(
        df_eig.style.format({
            "ส่วนจริง Re(λ)": "{:.6f}",
            "ส่วนจินตภาพ Im(λ)": "{:.6f}",
            "ขนาด |λ|": "{:.6f}"
        }),
        use_container_width=True
    )

