import streamlit as st
import time
import pandas as pd
import io
import re
import os
from datetime import datetime
from transformers import pipeline

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# ==============================
# 0. ตั้งค่าหน้าเว็บ (UI/UX)
# ==============================
st.set_page_config(
    page_title="FB Sentiment AI", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 1. โหลดโมเดล Sentiment Analysis
# ==============================
@st.cache_resource
def load_model():
    model_name = "firstmetis/Wisesight_finetune" 
    return pipeline("sentiment-analysis", model=model_name)

classifier = load_model()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# ==============================
# 2. Utility Functions (คงเดิม 100%)
# ==============================
def extract_text_with_emoji(driver, element):
    js = """
    function getText(node) {
        let text = '';
        node.childNodes.forEach(n => {
            if (n.nodeType === Node.TEXT_NODE) {
                text += n.textContent;
            } else if (n.nodeType === Node.ELEMENT_NODE) {
                if (n.tagName === 'IMG' && n.alt) {
                    text += n.alt;
                } else {
                    text += getText(n);
                }
            }
        });
        return text;
    }
    return getText(arguments[0]);
    """
    try:
        return driver.execute_script(js, element).strip()
    except:
        try: return element.text
        except: return ""

def auto_click_all_comments(driver):
    try:
        current_filter_xpath = "//div[@role='button']//span[text()='All comments' or text()='ความคิดเห็นทั้งหมด']"
        if driver.find_elements(By.XPATH, current_filter_xpath):
            return True

        trigger_xpath = "//div[@role='button']//span[text()='Most relevant' or text()='ความคิดเห็นที่เกี่ยวข้อง' or text()='Newest' or text()='ล่าสุด']"
        triggers = driver.find_elements(By.XPATH, trigger_xpath)
        
        clicked_dropdown = False
        for trigger in triggers:
            try:
                if trigger.is_displayed():
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", trigger)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", trigger)
                    clicked_dropdown = True
                    time.sleep(2) 
                    break
            except:
                continue
                
        if clicked_dropdown:
            exact_all_comments_xpath = "//div[@role='menuitem']//span[text()='All comments' or text()='ความคิดเห็นทั้งหมด']"
            options = driver.find_elements(By.XPATH, exact_all_comments_xpath)
            for opt in options:
                try:
                    if opt.is_displayed():
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", opt)
                        time.sleep(0.5)
                        driver.execute_script("arguments[0].click();", opt)
                        time.sleep(3) 
                        return True
                except: continue
    except Exception as e: pass
    return False

def find_scroll_container(driver, min_client_height=200):
    script = """
    const nodes = document.querySelectorAll('div');
    let best = null;
    for (const el of nodes) {
        try {
            const style = window.getComputedStyle(el);
            if (!el.offsetParent) continue;
            if (!/auto|scroll|overlay/i.test(style.overflowY || '')) continue;
            const sh = el.scrollHeight || 0;
            const ch = el.clientHeight || 0;
            if (sh > ch && ch >= arguments[0]) {
                if (!best || (sh - ch) > best.gap) {
                    best = { el, gap: sh - ch };
                }
            }
        } catch(e){}
    }
    return best ? best.el : null;
    """
    return driver.execute_script(script, min_client_height)

def brute_force_open_all(driver, rounds=6, sleep_time=1.2):
    js = """
    let clicked = 0;
    const keywords = [
        'ดูความคิดเห็น','ดูเพิ่มเติม','เพิ่มเติม','more comments','view more',
        'view all', 'view previous comments', 'more replies','replied','replies', 
        'ตอบกลับ', 'ข้อความตอบกลับ', 'reply'
    ].map(p => p.toLowerCase());
    
    document.querySelectorAll('span, div[role="button"], a').forEach(el => {
        const t = (el.innerText || '').toLowerCase();
        if (keywords.some(k => t.includes(k)) && el.offsetParent !== null) {
            try {
                el.scrollIntoView({block:'center'});
                el.click();
                clicked++;
            } catch(e){}
        }
    });
    return clicked;
    """
    total = 0
    for _ in range(rounds):
        try:
            c = driver.execute_script(js)
            total += c
            if c == 0: break
            time.sleep(sleep_time)
        except: break
    return total

# ==============================
# 3. Streamlit UI (ออกแบบใหม่)
# ==============================

# --- Sidebar แนะนำการใช้งาน ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Facebook_Logo_%282019%29.png/1024px-Facebook_Logo_%282019%29.png", width=60)
    st.title("📌 วิธีการใช้งาน")
    st.markdown("""
    1. ก๊อปปี้ลิงก์โพสต์ Facebook ที่ต้องการ (ต้องเป็นโพสต์สาธารณะ)
    2. วางลิงก์ลงในกล่องข้อความตรงกลางหน้าจอ 
       *(ใส่ได้หลายลิงก์ โดยเว้น 1 บรรทัดต่อ 1 ลิงก์)*
    3. กดปุ่ม **🚀 เริ่มดึงข้อมูล**
    4. รอระบบทำงาน (อาจใช้เวลาหลายนาทีหากคอมเมนต์มีจำนวนมาก)
    """)
    st.divider()
    st.caption("⚙️ **System:** ใช้ Web Scraping ร่วมกับ AI (Wisesight_finetune) เพื่อคัดกรองภาษาไทยและวิเคราะห์ Sentiment")

# --- ส่วนหัวหลัก (Header) ---
st.title("✨ FB Comment Scraper & Sentiment AI")
st.markdown("ดึงข้อความจากโพสต์ Facebook อัตโนมัติ พร้อมประเมินความรู้สึกของลูกค้าด้วย AI")
st.write("") # เว้นบรรทัดให้ดูโปร่ง

# --- กล่องรับ Input ---
with st.container(border=True):
    fb_urls_input = st.text_area(
        "🔗 วาง Link โพสต์ Facebook ที่นี่:", 
        height=120,
        placeholder="https://www.facebook.com/share/p/XXXXX/\nhttps://www.facebook.com/share/p/YYYYY/",
        key="unique_urls"
    )

    submit_btn = st.button("🚀 เริ่มดึงข้อมูลและวิเคราะห์", use_container_width=True, type="primary")

# ==============================
# 4. ระบบทำงานหลัก (Backend Process สำหรับ Deploy)
# ==============================
if submit_btn:
    urls = [url.strip() for url in fb_urls_input.split('\n') if url.strip()]
    
    if not urls:
        st.warning("⚠️ กรุณาใส่ลิงก์โพสต์ Facebook ก่อนครับ")
    else:
        ALL_COMMENT_DATA = [] 
        driver = None
        
        try:
            # ⭐️ อัปเดต Options สำหรับการ Deploy บน Server โดยเฉพาะ (Headless Mode)
            options = Options()
            options.add_argument("--headless=new") 
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-notifications")
            options.add_argument("--window-size=1920,1080")
            
            # # ให้ webdriver_manager จัดการโหลด Driver บน Server ให้อัตโนมัติโดยไม่ต้องอ้างอิงไดรฟ์ C:
            # service = Service(ChromeDriverManager().install())
            # driver = webdriver.Chrome(service=service, options=options)

            # ให้ webdriver_manager จัดการโหลด Driver บน Server ให้อัตโนมัติโดยไม่ต้องอ้างอิงไดรฟ์ C:
            if os.path.exists("/usr/bin/chromedriver"):
                options.binary_location = "/usr/bin/chromium"
                service = Service("/usr/bin/chromedriver")
            else:
                # ถ้ารันบนเครื่องตัวเอง (Windows/Mac) ค่อยให้ระบบโหลด Driver อัตโนมัติ
                service = Service(ChromeDriverManager().install())
                
            driver = webdriver.Chrome(service=service, options=options)
            
            # ⭐️ ใช้ st.status เพื่อรวม Log การทำงานไว้ในกล่องเดียว ไม่รกหน้าจอ
            with st.status("⚙️ บอทกำลังดึงข้อมูลจาก Facebook...", expanded=True) as status:
                
                for idx, current_url in enumerate(urls):
                    st.write(f"🌐 **โพสต์ที่ {idx + 1}/{len(urls)}:** เปิดลิงก์เป้าหมาย...")
                    driver.get(current_url)
                    
                    if idx == 0:
                        st.write("⏳ หากมี Popup Login ให้รีบกดปิดด้วยมือภายใน 5 วินาที...")
                        time.sleep(5)
                    else:
                        time.sleep(3) 

                    # --- ส่วนไถหน้าจอ ---
                    st.write(f"⏬ **โพสต์ที่ {idx + 1}:** กำลังไถหน้าจอและขยายคอมเมนต์...")
                    auto_click_all_comments(driver)
                    
                    max_scrolls = 200
                    no_progress_count = 0
                    
                    def click_load_more_main_comments():
                        js_click_more = """
                        let clicked = false;
                        const kws = ['view more comments', 'ดูความคิดเห็นเพิ่มเติม', 'view previous comments', 'ดูความคิดเห็นก่อนหน้า', 'ดูความคิดเห็นเพิ่มเติม...'];
                        document.querySelectorAll('span, div[role="button"]').forEach(el => {
                            const t = (el.innerText || '').toLowerCase().trim();
                            if (kws.some(k => t === k || t.includes(k)) && el.offsetParent !== null) {
                                try {
                                    el.scrollIntoView({block:'center'});
                                    el.click();
                                    clicked = true;
                                } catch(e){}
                            }
                        });
                        return clicked;
                        """
                        return driver.execute_script(js_click_more)
                    
                    scroll_div = find_scroll_container(driver, 300)
                    
                    if scroll_div:
                        last_height = driver.execute_script("return arguments[0].scrollHeight;", scroll_div)
                        for i in range(max_scrolls): 
                            try:
                                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", scroll_div)
                                time.sleep(2.5) 
                                new_height = driver.execute_script("return arguments[0].scrollHeight;", scroll_div)
                                
                                if new_height == last_height:
                                    if click_load_more_main_comments():
                                        time.sleep(3)
                                        no_progress_count = 0
                                        continue 
                                        
                                    time.sleep(3) 
                                    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", scroll_div)
                                    time.sleep(2)
                                    new_height = driver.execute_script("return arguments[0].scrollHeight;", scroll_div)
                                    if new_height == last_height:
                                        no_progress_count += 1
                                        if no_progress_count >= 4: break
                                    else: no_progress_count = 0
                                else: no_progress_count = 0
                                
                                last_height = new_height
                            except:
                                scroll_div = find_scroll_container(driver, 300)
                                if not scroll_div: break
                            if i % 6 == 0: brute_force_open_all(driver, rounds=1, sleep_time=1)
                    else:
                        last_height = driver.execute_script("return document.body.scrollHeight")
                        for i in range(max_scrolls):
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(2.5)
                            new_height = driver.execute_script("return document.body.scrollHeight")
                            if new_height == last_height:
                                if click_load_more_main_comments():
                                    time.sleep(3)
                                    no_progress_count = 0
                                    continue
                                time.sleep(3)
                                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                                time.sleep(2)
                                new_height = driver.execute_script("return document.body.scrollHeight")
                                if new_height == last_height:
                                    no_progress_count += 1
                                    if no_progress_count >= 4: break
                                else: no_progress_count = 0
                            else: no_progress_count = 0
                                
                            last_height = new_height
                            if i % 6 == 0: brute_force_open_all(driver, rounds=1, sleep_time=1)
                    
                    brute_force_open_all(driver, rounds=4, sleep_time=1.5)

                    # --- ส่วนดึงข้อความ ---
                    st.write(f"🔍 **โพสต์ที่ {idx + 1}:** กำลังกวาดข้อความและกรองแคปชัน...")
                    caption_blacklist = set()
                    try:
                        main_post_containers = driver.find_elements(By.CSS_SELECTOR, 'div[data-ad-comet-preview="message"]')
                        for container in main_post_containers:
                            blocks = container.find_elements(By.CSS_SELECTOR, 'div[dir="auto"]')
                            for b in blocks:
                                t = extract_text_with_emoji(driver, b).replace("\n", " ").strip()
                                if t: caption_blacklist.add(t)
                    except: pass
                        
                    all_text_blocks = driver.find_elements(By.CSS_SELECTOR, 'div[dir="auto"]')
                    try:
                        if all_text_blocks:
                            first_text = extract_text_with_emoji(driver, all_text_blocks[0]).replace("\n", " ").strip()
                            if first_text: caption_blacklist.add(first_text)
                    except: pass
                    
                    seen_texts = set() 
                    for block in all_text_blocks:
                        try:
                            text = extract_text_with_emoji(driver, block)
                            text = text.replace("\n", " ").strip()
                            if not text: continue
                            if text in caption_blacklist: continue
                            
                            ignore_exact = ['ถูกใจ', 'ตอบกลับ', 'แชร์', 'ซ่อน', 'ความคิดเห็น', 'View more', 'See more', 'ผู้ติดตาม', 'Top fan', 'ผู้เขียน', 'เขียนความคิดเห็น...', 'Write a comment...']
                            if any(w == text for w in ignore_exact): continue
                            if len(text) > 4000: continue
                            
                            has_thai = bool(re.search(r'[\u0E00-\u0E7F]', text))
                            has_english = bool(re.search(r'[a-zA-Z]', text))
                            if not has_thai and has_english: continue
                                
                            if text not in seen_texts:
                                seen_texts.add(text)
                                ALL_COMMENT_DATA.append({
                                    "post_url": current_url,
                                    "comment": text
                                })
                        except Exception: continue
                        
                    st.write(f"✅ **โพสต์ที่ {idx + 1}:** กวาดมาได้ {len(seen_texts)} คอมเมนต์")
                
                # พับกล่องสถานะเก็บเมื่อเสร็จสิ้น
                status.update(label="🎉 ดึงข้อมูลเสร็จสมบูรณ์! กำลังส่งต่อให้ AI...", state="complete", expanded=False)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
        finally:
            if driver:
                try: driver.quit()
                except: pass

        # ==============================
        # 5. วิเคราะห์ Sentiment & สร้าง Dashboard
        # ==============================
        if not ALL_COMMENT_DATA:
            st.error("❌ ไม่พบคอมเมนต์จากทุกลิงก์")
        else:
            df = pd.DataFrame(ALL_COMMENT_DATA)
            
            with st.spinner("🤖 AI กำลังประมวลผลความรู้สึก..."):
                texts_to_analyze = df["comment"].tolist()
                results = classifier(texts_to_analyze, truncation=True)
                
                df['sentiment'] = [label_map.get(res['label'], res['label']) for res in results]
                df['confidence_score'] = [round(res['score'], 4) for res in results]
            
            # --- Dashboard Section ---
            st.divider()
            st.subheader("📊 สรุปผลการวิเคราะห์ (Dashboard)")
            
            # คำนวณตัวเลข
            sentiment_counts = df['sentiment'].value_counts()
            pos_count = sentiment_counts.get('Positive', 0)
            neu_count = sentiment_counts.get('Neutral', 0)
            neg_count = sentiment_counts.get('Negative', 0)
            
            # จัดเรียง Metrics ให้สวยงาม
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💬 คอมเมนต์ทั้งหมด", f"{len(df)} รายการ")
            col2.metric("🟢 Positive (เชิงบวก)", pos_count)
            col3.metric("🟡 Neutral (เป็นกลาง)", neu_count)
            col4.metric("🔴 Negative (เชิงลบ)", neg_count)
            
            # แสดงกราฟแท่ง
            st.markdown("##### สัดส่วนความรู้สึกของคอมเมนต์")
            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [pos_count, neu_count, neg_count]
            }).set_index("Sentiment")
            st.bar_chart(chart_data)

            # ซ่อนตารางไว้ในกล่อง Expander เพื่อไม่ให้รก
            with st.expander("📄 ดูตารางข้อมูลดิบ (DataFrame)", expanded=True):
                st.dataframe(df, use_container_width=True)

            # ปุ่มดาวน์โหลด (เปลี่ยนชื่อไฟล์ตามที่ร้องขอ)
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False, encoding="utf-8-sig")

            st.download_button(
                label="📥 ดาวน์โหลดข้อมูลเป็นไฟล์ CSV",
                data=buffer.getvalue(),
                file_name=f"Receipt_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
