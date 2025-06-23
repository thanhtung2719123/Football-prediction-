import sys
import os
import re
import json
import pandas as pd
import numpy as np
import google.generativeai as genai
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QInputDialog, QMessageBox,
    QVBoxLayout, QWidget, QMenuBar, QTabWidget, QPushButton, QHBoxLayout,
    QDialog, QLineEdit, QFormLayout, QDialogButtonBox, QComboBox, QLabel, QGroupBox,
    QFileDialog, QProgressDialog
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt, QObject, QSize
)
import asyncio
from mplsoccer import Pitch

from football_scraper import (
    get_fotmob_match_data,
    get_transfermarkt_player_data,
    get_match_id_from_url, 
    scrape_fotmob_match, 
    scrape_sofascore_match,
    get_fotmob_team_recent_match_ids,
    get_sofascore_shotmap,
)

# --- Dialog for URL Input ---
class TwoMatchUrlDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nhập URL Trận đấu")
        self.layout = QFormLayout(self)

        self.match1_url_edit = QLineEdit(self)
        self.match2_url_edit = QLineEdit(self)
        self.match1_url_edit.setPlaceholderText("https://www.fotmob.com/match/...")
        self.match2_url_edit.setPlaceholderText("https://www.fotmob.com/match/...")

        self.layout.addRow("URL Trận 1:", self.match1_url_edit)
        self.layout.addRow("URL Trận 2:", self.match2_url_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.layout.addWidget(button_box)

    def get_urls(self):
        return self.match1_url_edit.text().strip(), self.match2_url_edit.text().strip()

# --- Dialog for Team Selection ---
class TeamSelectionDialog(QDialog):
    def __init__(self, team_options, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chọn 2 Đội để So Sánh")
        
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("Lấy dữ liệu thành công. Vui lòng chọn 2 đội bạn muốn phân tích:"))
        
        self.team1_combo = QComboBox(self)
        self.team2_combo = QComboBox(self)
        
        # Ensure unique teams are added to the dropdown
        for team_id, team_name in team_options.items():
            self.team1_combo.addItem(team_name, userData=team_id)
            self.team2_combo.addItem(team_name, userData=team_id)
            
        self.layout.addWidget(QLabel("Đội 1 (Coi là đội nhà):"))
        self.layout.addWidget(self.team1_combo)
        self.layout.addWidget(QLabel("Đội 2 (Coi là đội khách):"))
        self.layout.addWidget(self.team2_combo)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        self.layout.addWidget(button_box)

    def get_selection(self):
        team1_id = self.team1_combo.currentData()
        team1_name = self.team1_combo.currentText()
        team2_id = self.team2_combo.currentData()
        team2_name = self.team2_combo.currentText()
        
        if team1_id == team2_id:
            return None, None, None, None # Indicate error
        
        return team1_id, team1_name, team2_id, team2_name

# --- Worker Threads ---
class MultiMatchScraperWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, match1_url, match2_url):
        super().__init__()
        self.match1_url = match1_url
        self.match2_url = match2_url

    def run(self):
        try:
            match1_data = get_fotmob_match_data(self.match1_url)
            match2_data = get_fotmob_match_data(self.match2_url)
            
            if not match1_data.get('team_data') or not match2_data.get('team_data'):
                 self.finished.emit(None, "Không thể lấy dữ liệu đội từ một hoặc cả hai trận đấu.")
                 return

            all_teams_for_selection = {**match1_data['team_data'], **match2_data['team_data']}
            
            # Keep match data separate but provide a combined list of teams for the selection dialog
            processed_data = {
                'match1': match1_data,
                'match2': match2_data,
                'all_teams_for_selection': all_teams_for_selection
            }

            self.finished.emit(processed_data, None)
        except Exception as e:
            self.finished.emit(None, str(e))

class SingleScraperWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, fn, url):
        super().__init__()
        self.fn = fn
        self.url = url

    def run(self):
        try:
            data = self.fn(self.url)
            self.finished.emit(data, None)
        except Exception as e:
            self.finished.emit(None, str(e))

class Worker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, data, odds, gemini_api_key, parent=None):
        super().__init__(parent)
        self.data = data
        self.odds = odds
        self.gemini_api_key = gemini_api_key

    def run(self):
        try:
            if not self.gemini_api_key:
                raise ValueError("Vui lòng nhập API Key của Gemini trong menu 'Cài đặt'.")

            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('models/gemini-2.5-flash')

            # The 'data' dictionary now contains the detailed, formatted stats summaries
            home_team_name = self.data.get("home_team_name", "Đội nhà")
            away_team_name = self.data.get("away_team_name", "Đội khách")
            home_stats_summary = self.data.get("home_team_stats_summary", "Không có dữ liệu.")
            away_stats_summary = self.data.get("away_team_stats_summary", "Không có dữ liệu.")

            odds_euro_home = self.odds.get('euro', {}).get('home') or "N/A"
            odds_euro_draw = self.odds.get('euro', {}).get('draw') or "N/A"
            odds_euro_away = self.odds.get('euro', {}).get('away') or "N/A"
            odds_handicap_line = self.odds.get('handicap', {}).get('line') or "N/A"
            odds_handicap_home = self.odds.get('handicap', {}).get('home') or "N/A"
            odds_handicap_away = self.odds.get('handicap', {}).get('away') or "N/A"
            odds_ou_line = self.odds.get('ou', {}).get('line') or "N/A"
            odds_ou_over = self.odds.get('ou', {}).get('over') or "N/A"
            odds_ou_under = self.odds.get('ou', {}).get('under') or "N/A"

            prompt = f"""
            **YÊU CẦU:**
            Bạn là một chuyên gia phân tích bóng đá. Dựa trên dữ liệu và kèo nhà cái, hãy cung cấp một JSON object chứa các dự đoán và một bài phân tích chi tiết.

            **QUAN TRỌNG:** Luôn bắt đầu câu trả lời của bạn bằng một JSON object hợp lệ, theo sau là dấu phân cách "---" và sau đó là bài phân tích bằng văn bản.

            **ĐỊNH DẠNG JSON:**
            {{
              "prediction": {{
                "home_team_win_prob_pct": <số nguyên, xác suất thắng của đội nhà (0-100)>,
                "draw_prob_pct": <số nguyên, xác suất hòa (0-100)>,
                "away_team_win_prob_pct": <số nguyên, xác suất thắng của đội khách (0-100)>,
                "expected_total_goals": <số thực, tổng số bàn thắng kỳ vọng trong trận đấu>,
                "best_bet": "<lựa chọn kèo tốt nhất, ví dụ: {home_team_name} -0.5>",
                "confidence_level": "<'High' | 'Medium' | 'Low', mức độ tự tin cho 'best_bet'>",
                "score_probabilities": [
                  {{ "score": "<dự đoán tỷ số 1, ví dụ: '2-1'>", "probability_pct": <số nguyên, xác suất cho tỷ số đó> }},
                  {{ "score": "<dự đoán tỷ số 2, ví dụ: '1-1'>", "probability_pct": <số nguyên> }},
                  {{ "score": "<dự đoán tỷ số 3, ví dụ: '1-0'>", "probability_pct": <số nguyên> }}
                ]
              }}
            }}
            ---
            **BÀI PHÂN TÍCH TRẬN ĐẤU**

            **TRẬN ĐẤU:** {home_team_name} vs {away_team_name}

            **DỮ LIỆU TỔNG QUAN:**
            - **{home_team_name}:**
            {home_stats_summary}
            - **{away_team_name}:**
            {away_stats_summary}
            
            **Tỷ lệ kèo nhà cái:**
            - **Kèo Châu Âu (1x2):** Thắng: {odds_euro_home} | Hòa: {odds_euro_draw} | Thua: {odds_euro_away}
            - **Kèo Châu Á (Handicap):** Kèo: {odds_handicap_line}
            - **Kèo Tài Xỉu (O/U):** Mốc: {odds_ou_line}

            **PHÂN TÍCH:**
            1.  **Phân tích Phong độ & BXH:** Dựa vào dữ liệu thống kê, lịch sử đối đầu và vị trí trên bảng xếp hạng, đội nào có lợi thế?
            2.  **Phân tích Lối chơi & Đội hình:** Sơ đồ chiến thuật và các cầu thủ ra sân (nếu có) tiết lộ gì về lối chơi của họ?
            3.  **Phân tích Kèo:** So sánh nhận định của bạn với kèo nhà cái. Kèo có hợp lý không?
            4.  **Kết luận & Lựa chọn Tốt nhất:** Tóm tắt nhận định và giải thích tại sao bạn lại đưa ra lựa chọn kèo ở trong phần JSON.
            5.  **Dự đoán tỷ số:**
            """
            
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.7))
            self.finished.emit(response.text)
        except Exception as e:
            self.error.emit(str(e))

    def format_matches(self, matches):
        # This function is no longer used in the new workflow but kept for now.
        text = ""
        if not matches:
            return "Không có dữ liệu trận đấu.\n"
        for i, match in enumerate(matches):
            stats = match.get('stats', {})
            total_shots_stats = stats.get('Tất cả các cú sút', {})
            goals = total_shots_stats.get('Goals', 'N/A')
            xg = total_shots_stats.get('Expected goals (xG)', 'N/A')
            
            overview_stats = stats.get('Tổng quan', {})
            possession = overview_stats.get('Ball possession', 'N/A')
            
            text += f"- Trận vs {match['opponent_name']}: Ghi {goals} bàn, Sút (xG: {xg}), Kiểm soát bóng: {possession}\n"
        return text

def draw_pitch(ax):
    """Draws a football pitch on a matplotlib axes."""
    ax.add_patch(plt.Rectangle((0, 0), 120, 80, facecolor='#228B22', edgecolor='white', lw=2))
    ax.plot([60, 60], [0, 80], color="white", lw=2)
    ax.add_patch(plt.Circle((60, 40), 9.15, ec='white', fc='none', lw=2))
    ax.add_patch(plt.Rectangle((0, 18), 16.5, 44, ec='white', fc='none', lw=2))
    ax.add_patch(plt.Rectangle((103.5, 18), 16.5, 44, ec='white', fc='none', lw=2))
    ax.add_patch(plt.Rectangle((0, 30), 5.5, 20, ec='white', fc='none', lw=2))
    ax.add_patch(plt.Rectangle((114.5, 30), 5.5, 20, ec='white', fc='none', lw=2))
    ax.add_patch(plt.Circle((12, 40), 0.8, color="white"))
    ax.add_patch(plt.Circle((108, 40), 0.8, color="white"))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

class OddsInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nhập Kèo Nhà Cái")
        self.layout = QFormLayout(self)

        self.home_euro = QLineEdit(self)
        self.draw_euro = QLineEdit(self)
        self.away_euro = QLineEdit(self)
        self.handicap_line = QLineEdit(self)
        self.home_handicap = QLineEdit(self)
        self.away_handicap = QLineEdit(self)
        self.ou_line = QLineEdit(self)
        self.over_ou = QLineEdit(self)
        self.under_ou = QLineEdit(self)

        self.layout.addRow("Kèo 1x2 - Thắng:", self.home_euro)
        self.layout.addRow("Kèo 1x2 - Hòa:", self.draw_euro)
        self.layout.addRow("Kèo 1x2 - Thua:", self.away_euro)
        self.layout.addRow("Kèo Châu Á - Kèo:", self.handicap_line)
        self.layout.addRow("Kèo Châu Á - Tỷ lệ đội nhà:", self.home_handicap)
        self.layout.addRow("Kèo Châu Á - Tỷ lệ đội khách:", self.away_handicap)
        self.layout.addRow("Kèo Tài Xỉu - Mốc:", self.ou_line)
        self.layout.addRow("Kèo Tài Xỉu - Tài:", self.over_ou)
        self.layout.addRow("Kèo Tài Xỉu - Xỉu:", self.under_ou)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.layout.addWidget(button_box)

    def get_odds(self):
        return {
            "euro": {
                "home": self.home_euro.text(),
                "draw": self.draw_euro.text(),
                "away": self.away_euro.text(),
            },
            "handicap": {
                "line": self.handicap_line.text(),
                "home": self.home_handicap.text(),
                "away": self.away_handicap.text(),
            },
            "ou": {
                "line": self.ou_line.text(),
                "over": self.over_ou.text(),
                "under": self.under_ou.text(),
            }
        }

# --- Main Application ---
class ScraperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("kèo bóng VTT")
        self.setGeometry(100, 100, 1200, 800)

        # --- Main Layout ---
        self.tabs = QTabWidget()
        self.raw_data_text = QTextEdit()
        self.raw_data_text.setReadOnly(True)
        self.ai_analysis_text = QTextEdit()
        self.ai_analysis_text.setReadOnly(True)
        
        self.ai_vis_tab = QWidget()
        self.ai_vis_layout = QHBoxLayout(self.ai_vis_tab)
        self.shotmap_history_tab = QWidget()
        self.shotmap_layout = QHBoxLayout(self.shotmap_history_tab)

        self.tabs.addTab(self.raw_data_text, "Dữ liệu thô")
        self.tabs.addTab(self.ai_analysis_text, "Phân tích AI")
        self.tabs.addTab(self.ai_vis_tab, "Trực quan hóa AI")
        self.tabs.addTab(self.shotmap_history_tab, "Shotmap Lịch sử")
        
        self.setCentralWidget(self.tabs)
        
        self._create_menu_bar()
        self.create_progress_dialog()

        # --- App State ---
        self.raw_data = None
        self.current_config = None
        self.gemini_api_key = None
        self.thread = QThread() # Luồng cho AI worker
        self.ai_prediction_data = None # Store AI prediction JSON

    def _create_menu_bar(self):
        menubar = self.menuBar()
        
        # --- File Menu ---
        file_menu = menubar.addMenu("Hành động")
        start_action = QAction("Bắt đầu Phân tích Mới", self)
        start_action.triggered.connect(self.start_analysis)
        file_menu.addAction(start_action)
        
        # --- Settings Menu ---
        settings_menu = menubar.addMenu("Cài đặt")
        api_key_action = QAction("Nhập Gemini API Key", self)
        api_key_action.triggered.connect(self.set_api_key)
        settings_menu.addAction(api_key_action)

    def create_progress_dialog(self):
        self.progress_dialog = QProgressDialog("Đang xử lý...", "Hủy", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.setCancelButton(None) # Không cho phép hủy
        self.progress_dialog.hide()
        
    def update_progress_dialog(self, message, value):
        self.progress_dialog.setLabelText(message)
        self.progress_dialog.setValue(value)

    def start_analysis(self):
        """Gets two match URLs from the user and starts the scraping process."""
        if not self.gemini_api_key:
            self.set_api_key()
            if not self.gemini_api_key: 
                return 

        dialog = TwoMatchUrlDialog(self)
        if dialog.exec():
            match1_url, match2_url = dialog.get_urls()

            if not match1_url or not match2_url or not "fotmob.com" in match1_url or not "fotmob.com" in match2_url:
                QMessageBox.warning(self, "URL không hợp lệ", "Vui lòng nhập hai URL hợp lệ từ FotMob.")
                return

            # Clear previous results
            self.raw_data_text.clear()
            self.ai_analysis_text.clear()
            for i in reversed(range(self.ai_vis_layout.count())): 
                self.ai_vis_layout.itemAt(i).widget().setParent(None)
            for i in reversed(range(self.shotmap_layout.count())): 
                self.shotmap_layout.itemAt(i).widget().setParent(None)

            self.progress_dialog.setLabelText("Đang cào dữ liệu từ 2 trận đấu...")
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()
            
            self.scraper_worker = MultiMatchScraperWorker(match1_url, match2_url)
            self.scraper_worker.finished.connect(self.on_scraping_finished)
            self.scraper_worker.start()

    def on_scraping_finished(self, data, error_message):
        self.progress_dialog.hide()
        if error_message:
            self.on_scraping_error(error_message)
            return
            
        if not data or not data.get('all_teams_for_selection'):
            QMessageBox.critical(self, "Lỗi", "Không thể lấy dữ liệu từ các trận đấu đã cho.")
            return

        self.raw_data = data # Store combined data
        
        # --- Team Selection ---
        team_options = self.raw_data['all_teams_for_selection']
        dialog = TeamSelectionDialog(team_options, self)
        if dialog.exec():
            team1_id, team1_name, team2_id, team2_name = dialog.get_selection()
            if not team1_id or not team2_id:
                QMessageBox.warning(self, "Lỗi", "Lựa chọn đội không hợp lệ hoặc bạn đã chọn cùng một đội hai lần.")
                return
            
            self.prepare_and_run_ai_analysis(team1_id, team1_name, team2_id, team2_name)
        else:
            # User cancelled team selection
            self.raw_data_text.setText("Phân tích đã bị hủy vì chưa chọn đội.")

    def on_scraping_error(self, message):
        self.progress_dialog.hide()
        QMessageBox.critical(self, "Lỗi cào dữ liệu", message)

    def prepare_and_run_ai_analysis(self, home_team_id, home_team_name, away_team_id, away_team_name):
        """Filters the combined data for the selected teams and runs the AI."""
        
        # Store selected teams for visualization later
        self.selected_home_team_info = {'id': home_team_id, 'name': home_team_name}
        self.selected_away_team_info = {'id': away_team_id, 'name': away_team_name}

        self.ai_analysis_text.setText("Đang chuẩn bị dữ liệu và gửi tới AI...")
        self.tabs.setCurrentWidget(self.ai_analysis_text)

        # --- Find the correct match data for each selected team ---
        home_match_data = None
        if home_team_id in self.raw_data['match1']['team_data']:
            home_match_data = self.raw_data['match1']
        elif home_team_id in self.raw_data['match2']['team_data']:
            home_match_data = self.raw_data['match2']

        away_match_data = None
        if away_team_id in self.raw_data['match1']['team_data']:
            away_match_data = self.raw_data['match1']
        elif away_team_id in self.raw_data['match2']['team_data']:
            away_match_data = self.raw_data['match2']
            
        if not home_match_data or not away_match_data:
            QMessageBox.critical(self, "Lỗi Dữ liệu", "Không thể tìm thấy dữ liệu trận đấu cho các đội đã chọn.")
            return
            
        # --- Format data for the AI ---
        home_stats_summary = self.format_full_data_for_ai(home_match_data.get('full_data', {}), home_team_name)
        away_stats_summary = self.format_full_data_for_ai(away_match_data.get('full_data', {}), away_team_name)

        # We need to get the odds from the user. For now, let's use a dummy dialog.
        odds_dialog = OddsInputDialog(self)
        if not odds_dialog.exec():
            self.ai_analysis_text.setText("Phân tích đã bị hủy vì chưa nhập kèo.")
            return
            
        odds = odds_dialog.get_odds()

        # Create a simplified data structure for the AI
        analysis_data = {
            "home_team_name": home_team_name,
            "away_team_name": away_team_name,
            "home_team_stats_summary": home_stats_summary,
            "away_team_stats_summary": away_stats_summary,
        }
        
        # Combine shots from both matches for the shotmap tab later
        combined_shots_df = pd.concat(
            [self.raw_data['match1']['shots_df'], self.raw_data['match2']['shots_df']], 
            ignore_index=True
        )
        self.processed_shots_df_for_vis = combined_shots_df

        raw_display_text = {
            f"Phân tích cho": f"{home_team_name} vs {away_team_name}",
            f"Dữ liệu của {home_team_name} được lấy từ trận đấu của họ trong bộ dữ liệu.": home_stats_summary,
            f"Dữ liệu của {away_team_name} được lấy từ trận đấu của họ trong bộ dữ liệu.": away_stats_summary,
            "Kèo đã nhập": odds
        }
        self.raw_data_text.setPlainText(json.dumps(raw_display_text, indent=2, ensure_ascii=False))
        
        self.ai_prediction_data = None # Reset previous AI data
        self.worker = Worker(analysis_data, odds, self.gemini_api_key)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_ai_finished)
        self.worker.error.connect(self.on_ai_error)
        self.thread.start()

    def update_tabs_with_new_data(self):
        # This function is now mostly obsolete and replaced by the new flow
        # We will call the AI worker directly from on_scraping_finished
        pass

    def on_ai_finished(self, result):
        try:
            # Split the response into JSON and text parts
            if "---" in result:
                json_str, analysis_text = result.split("---", 1)
            else: # Fallback if AI forgets the separator
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if not json_match:
                    raise ValueError("Không tìm thấy JSON hợp lệ trong phản hồi của AI.")
                json_str = json_match.group(0)
                analysis_text = result.replace(json_str, '')

            # Clean and parse the JSON
            json_str = json_str.strip().replace("```json", "").replace("```", "")
            ai_data = json.loads(json_str)
            
            self.ai_analysis_text.setPlainText(analysis_text.strip())
            self.ai_prediction_data = ai_data # Store for visualization
            
        except (ValueError, json.JSONDecodeError) as e:
            error_message = f"Lỗi khi xử lý phản hồi từ AI:\n{e}\n\nPhản hồi gốc:\n{result}"
            self.ai_analysis_text.setPlainText(error_message)
            self.ai_prediction_data = None
        
        self.thread.quit()
        self.thread.wait()
        self.update_visualization_tabs_after_ai()

    def on_ai_error(self, message):
        self.ai_analysis_text.setPlainText(f"Lỗi khi phân tích AI:\n{message}")
        self.ai_prediction_data = None
        self.thread.quit()
        self.thread.wait()

    def update_visualization_tabs_after_ai(self):
        # Clear existing visualizations
        for i in reversed(range(self.ai_vis_layout.count())): 
            self.ai_vis_layout.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.shotmap_layout.count())): 
            self.shotmap_layout.itemAt(i).widget().setParent(None)

        # --- AI Visualization Tab ---
        if hasattr(self, 'ai_prediction_data') and self.ai_prediction_data:
            try:
                prediction = self.ai_prediction_data.get('prediction', {})
                win_prob_chart = self.create_win_prob_chart(prediction)
                other_preds_display = self.create_other_predictions_display(prediction)
                
                # Add to a container widget for better layout control
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.addWidget(win_prob_chart)
                container_layout.addWidget(other_preds_display)
                container_layout.addStretch()

                self.ai_vis_layout.addWidget(container)

            except Exception as e:
                self.ai_vis_layout.addWidget(QLabel(f"Lỗi khi tạo trực quan hóa AI: {e}"))
        else:
            self.ai_vis_layout.addWidget(QLabel("Không có dữ liệu dự đoán từ AI để trực quan hóa."))


        # --- Shotmap Tab ---
        if hasattr(self, 'processed_shots_df_for_vis') and not self.processed_shots_df_for_vis.empty:
            shots_df = self.processed_shots_df_for_vis

            home_id = self.selected_home_team_info['id']
            away_id = self.selected_away_team_info['id']
            home_name = self.selected_home_team_info['name']
            away_name = self.selected_away_team_info['name']

            home_shots_df = shots_df[shots_df['teamId'] == home_id]
            away_shots_df = shots_df[shots_df['teamId'] == away_id]
            
            self.update_shotmap_tab_combined(home_shots_df, home_name, "Dữ liệu trận gần nhất")
            self.update_shotmap_tab_combined(away_shots_df, away_name, "Dữ liệu trận gần nhất")
        else:
             self.shotmap_layout.addWidget(QLabel("Không tìm thấy dữ liệu shotmap hoặc đội được chọn."))

    def create_win_prob_chart(self, prediction_data):
        fig = Figure(figsize=(6, 4), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        home_prob = prediction_data.get('home_team_win_prob_pct', 0)
        draw_prob = prediction_data.get('draw_prob_pct', 0)
        away_prob = prediction_data.get('away_team_win_prob_pct', 0)
        
        teams = [self.selected_away_team_info['name'], 'Hòa', self.selected_home_team_info['name']]
        probs = [away_prob, draw_prob, home_prob]
        colors = ['#d9534f', '#f0ad4e', '#5cb85c'] # Red, Orange, Green

        bars = ax.barh(teams, probs, color=colors, height=0.6)
        ax.set_xlabel('Xác suất (%)', color='white', fontsize=12)
        ax.set_title('Dự đoán Kết quả Trận đấu', color='white', fontsize=16)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white', labelsize=12)
        ax.set_xlim(0, 100)

        # Add percentage labels on bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 1
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width}%', ha='left', va='center', color='white', fontsize=11, fontweight='bold')

        fig.set_facecolor("#22312b")
        ax.set_facecolor("#22312b")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        
        fig.tight_layout()
        return canvas

    def create_other_predictions_display(self, prediction_data):
        widget = QGroupBox("Dự đoán Chi tiết")
        layout = QFormLayout()

        expected_goals = prediction_data.get('expected_total_goals', 'N/A')
        best_bet = prediction_data.get('best_bet', 'N/A')
        confidence = prediction_data.get('confidence_level', 'N/A')
        score_probs = prediction_data.get('score_probabilities', [])

        layout.addRow(QLabel("Tổng số bàn thắng kỳ vọng:"), QLabel(f"<b>{expected_goals}</b>"))
        layout.addRow(QLabel("Lựa chọn Tốt nhất (Best Bet):"), QLabel(f"<b>{best_bet}</b>"))
        layout.addRow(QLabel("Mức độ tự tin:"), QLabel(f"<b>{confidence}</b>"))
        
        # Add a separator
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #5A6A72;")
        layout.addRow(separator)
        
        # Display score probabilities
        if score_probs:
            # Use <br> for newlines in a QLabel with rich text
            scores_text = "<br>".join([f"<b>{item.get('score', '?')}</b>: {item.get('probability_pct', '?')}%" for item in score_probs])
            scores_label = QLabel(scores_text)
            scores_label.setWordWrap(True)
            layout.addRow(QLabel("Dự đoán Tỷ số:"), scores_label)
        
        widget.setLayout(layout)
        
        widget.setStyleSheet("""
            QGroupBox {
                background-color: #2E4045;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #5A6A72;
                border-radius: 8px;
                margin-top: 1ex;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: -12px 10px 0 10px;
                color: #E0E0E0;
            }
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: normal;
                background: transparent;
            }
        """)
        return widget

    def format_full_data_for_ai(self, full_data, team_name):
        """Formats all the new, detailed data into a string for the AI prompt."""
        if not full_data:
            return "Không có dữ liệu chi tiết."
    
        output = ""
        
        # 1. Goalscorers
        facts = full_data.get('matchFacts', {})
        if facts and 'goals' in facts and facts['goals']:
            output += "Ghi bàn:\n"
            for goal in facts['goals']:
                scorer_line = f"- {goal.get('scorerName', 'N/A')} {goal.get('timeStr', '')}"
                if goal.get('isOwnGoal'):
                    scorer_line += f" (Phản lưới nhà)"
                output += scorer_line + "\n"
        
        # 2. Detailed Stats
        stats_data = full_data.get('stats', {})
        if stats_data and 'stats' in stats_data and stats_data.get('stats'):
            output += "\nThống kê chi tiết:\n"
            for section in stats_data['stats']:
                if 'teamNames' not in section or len(section['teamNames']) < 2:
                    continue
                
                output += f"**{section['title']}**\n"
                for stat in section['stats']:
                    home_val = stat['stats'][0]
                    away_val = stat['stats'][1]
                    
                    team_val = "N/A"
                    if section['teamNames'][0] == team_name:
                        team_val = home_val
                    elif section['teamNames'][1] == team_name:
                        team_val = away_val
                    
                    output += f"- {stat['key']}: {team_val}\n"
        
        # 3. Lineup
        lineup_data = full_data.get('lineup', {})
        if lineup_data and lineup_data.get('lineup'):
            output += "\nĐội hình ra sân:\n"
            for team_lineup in lineup_data['lineup']:
                if team_lineup.get('teamName') == team_name:
                    output += f"- Sơ đồ: {team_lineup.get('formation', 'N/A')}\n"
        
        # 4. H2H
        h2h_data = full_data.get('h2h', {})
        if h2h_data and h2h_data.get('matches'):
            output += "\nLịch sử đối đầu (3 trận gần nhất):\n"
            for match in h2h_data['matches'][:3]:
                home = match.get('home', {}).get('name')
                away = match.get('away', {}).get('name')
                winner = match.get('winner')
                result = f"{home} {match.get('score')} {away}"
                if winner == 'home':
                    result += f" (Thắng: {home})"
                elif winner == 'away':
                    result += f" (Thắng: {away})"
                else:
                    result += " (Hòa)"
                output += f"- {result}\n"

        # 5. Table position
        table_data = full_data.get('table', {})
        if table_data:
            all_tables = table_data.get('tables', [])
            if all_tables:
                first_table = all_tables[0].get('table', {})
                if 'all' in first_table:
                    output += "\nBảng xếp hạng:\n"
                    for team_stats in first_table['all']:
                        if team_stats.get('name') == team_name:
                            pos = team_stats.get('idx')
                            pts = team_stats.get('pts')
                            played = team_stats.get('played')
                            wins = team_stats.get('wins')
                            draws = team_stats.get('draws')
                            losses = team_stats.get('losses')
                            gd = team_stats.get('goalDifference')
                            output += f"- Vị trí: {pos}, Điểm: {pts}, (Thắng: {wins}, Hòa: {draws}, Thua: {losses}), Hiệu số: {gd}\n"

        return output.strip() if output else "Không có dữ liệu chi tiết."

    def update_shotmap_tab_combined(self, shots_df, team_name, title_suffix):
        fig = Figure(figsize=(8, 5), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Use Opta pitch type which is 100x100, matching FotMob's coordinate system
        pitch = Pitch(pitch_type='opta', pitch_color='#22312b', line_color='#c7d5cc')
        pitch.draw(ax=ax)

        # The input is now a DataFrame, so we convert it to a list of dicts to iterate
        all_shots = shots_df.to_dict('records')

        if not all_shots:
            ax.text(60, 40, 'Không có dữ liệu shotmap', ha='center', va='center', fontsize=12, color='white')
        else:
            for shot in all_shots:
                x = shot['x']
                y = shot['y']
                is_goal = shot.get('eventType') == 'Goal'
                marker = '*' if is_goal else 'o'
                color = 'yellow' if is_goal else 'red'
                size = 200 if is_goal else 70
                pitch.scatter(x, y, ax=ax, s=size, c=color, marker=marker, edgecolors='black', alpha=0.9)

        ax.set_title(f"Tổng hợp Shotmap của {team_name}\n({title_suffix})", color="white", fontsize=14)
        fig.set_facecolor("#22312b")
        
        self.shotmap_layout.addWidget(canvas)

    def set_api_key(self):
        text, ok = QInputDialog.getText(self, 'Nhập API Key', 'Vui lòng nhập Google Gemini API Key của bạn:')
        if ok and text:
            self.gemini_api_key = text
            QMessageBox.information(self, "Thành công", "Đã lưu API Key cho phiên này.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ScraperApp()
    window.show()
    sys.exit(app.exec())
