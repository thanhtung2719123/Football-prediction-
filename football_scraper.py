import re
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


def get_fotmob_match_data(match_url: str) -> dict:
    """
    Scrapes shotmap and team data from a single FotMob match page.
    This is the primary function for fetching data for analysis.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            print(f"Navigating to match: {match_url}")
            page.goto(match_url, wait_until="domcontentloaded", timeout=60000)

            page_data_str = page.locator('script#__NEXT_DATA__').inner_text(timeout=15000)
            data = json.loads(page_data_str)
            
            general_props = data.get('props', {}).get('pageProps', {}).get('general', {})
            if not general_props:
                raise ValueError("Could not find 'general' properties in page data.")

            match_id = general_props.get('matchId')
            if not match_id:
                raise ValueError("Could not determine match ID from page data.")
            
            content_props = data.get('props', {}).get('pageProps', {}).get('content', {})
            if not content_props:
                raise ValueError("Could not find 'content' properties in page data.")

            # Extract data from all relevant tabs
            shots_list = content_props.get('shotmap', {}).get('shots', [])
            shots_df = pd.DataFrame(shots_list) if shots_list else pd.DataFrame()
            stats_data = content_props.get('stats', {})
            match_facts = content_props.get('matchFacts', {})
            lineup_data = content_props.get('lineup', {})
            h2h_data = content_props.get('h2h', {})
            table_data = data.get('props', {}).get('pageProps', {}).get('tableData', {})

            full_data = {
                "stats": stats_data,
                "matchFacts": match_facts,
                "lineup": lineup_data,
                "h2h": h2h_data,
                "table": table_data
            }
            
            home_team_data = general_props.get('homeTeam', {})
            away_team_data = general_props.get('awayTeam', {})
            
            team_data = {
                home_team_data.get('id'): home_team_data.get('name'),
                away_team_data.get('id'): away_team_data.get('name'),
            }
            # Clean out any entries where ID or name might be missing
            team_data = {k: v for k, v in team_data.items() if k and v}

            print(f"  - Successfully scraped match {match_id}. Found {len(shots_df)} shots.")
            return {'shots_df': shots_df, 'team_data': team_data, 'shotmap': shots_list, 'full_data': full_data}

        except Exception as e:
            print(f"  - Could not scrape match {match_url}. Reason: {e}")
            return {'shots_df': pd.DataFrame(), 'team_data': {}, 'shotmap': [], 'full_data': {}}
        finally:
            browser.close()


def _safe_regex_search(pattern: str, text: str) -> str:
    """Helper function to safely perform a regex search, returning 'N/A' if not found."""
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Handle cases where the main group is optional
        return (match.group(1) or 'N/A').strip()
    return "N/A"


def get_transfermarkt_player_data(player_url: str) -> dict:
    """
    Scrapes player data from Transfermarkt, including profile and market value history.

    Args:
        player_url: The URL of the player's profile on transfermarkt.us.
                    Example: "https://www.transfermarkt.us/erling-haaland/profil/spieler/418560"

    Returns:
        A dictionary containing various pieces of player data.
    """
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    }
    player_id = player_url.split('/')[-1]
    
    try:
        response = requests.get(player_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        player_data = {
            "player_id": player_id,
            "player_name": soup.select_one('h1.data-header__headline-wrapper').text.split('\\n')[-1].strip(),
            "player_number": (soup.select_one('span.data-header__shirt-number').text.strip().replace('#', '')) if soup.select_one('span.data-header__shirt-number') else "N/A"
        }

        # Use the safe regex helper to prevent crashes on missing data
        soup_text = soup.text
        player_data["contract_expiry"] = _safe_regex_search(r"Contract expires: (.*)", soup_text)
        player_data["birthplace"] = _safe_regex_search(r"Place of birth:.*?([A-z\s]+\,)", soup_text).rstrip(',')
        player_data["agent"] = _safe_regex_search(r"Agent:.*?([A-z\s\./-]+?)\n", soup_text)
        player_data["height"] = _safe_regex_search(r"Height:.*?(\d,\d{2}m)", soup_text)

        # Fetching API data
        player_data["market_value_history"] = _get_transfermarkt_api_data(player_id, "marketValueDevelopment/graph", headers)
        player_data["transfer_history"] = _get_transfermarkt_api_data(player_id, "transferHistory/list", headers)
        player_data["performance_data"] = _get_transfermarkt_api_data(player_id, f"player/{player_id}/performance", headers, is_player_performance=True)

        return player_data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Transfermarkt: {e}")
        return {}
    except (AttributeError, IndexError, TypeError) as e:
        print(f"Error parsing Transfermarkt page: {e}")
        return {}


def _get_transfermarkt_api_data(player_id: str, endpoint: str, headers: dict, is_player_performance: bool = False) -> dict:
    """Helper function to fetch data from Transfermarkt's API."""
    base_url = 'https://www.transfermarkt.us/ceapi/'
    url = f"{base_url}{endpoint}" if is_player_performance else f"{base_url}{endpoint}/{player_id}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching API data from {url}: {e}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {url}")
        return {}


def get_sofascore_shotmap(match_url: str) -> dict:
    """
    Scrapes shotmap and team data from a SofaScore match.
    Returns a dictionary with 'shots' (DataFrame) and 'teams' (dict).
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            print("Navigating to SofaScore page...")
            page.goto(match_url, wait_until="domcontentloaded", timeout=30000)

            # Handle cookie consent banners
            try:
                consent_button = page.locator('button:has-text("AGREE")').first
                if consent_button.is_visible(timeout=5000):
                    print("Cookie consent button found, clicking it.")
                    consent_button.click()
            except PlaywrightTimeoutError:
                print("No cookie consent button found, continuing.")

            print("Attempting to find event ID in embedded page data...")
            page_data_str = page.locator('script#__NEXT_DATA__').inner_text(timeout=7000)
            data = json.loads(page_data_str)
            
            # The numeric event ID is stored deep in the page's data structure.
            event_id = data.get('props', {}).get('pageProps', {}).get('event', {}).get('id')

            if not event_id:
                raise ValueError("Could not find event ID in page's embedded __NEXT_DATA__.")
            
            print(f"Successfully found event ID: {event_id}")

            # --- Extract Teams from Page Data ---
            home_team_data = data.get('props', {}).get('pageProps', {}).get('event', {}).get('homeTeam', {})
            away_team_data = data.get('props', {}).get('pageProps', {}).get('event', {}).get('awayTeam', {})
            teams_map = {
                home_team_data.get('id'): home_team_data.get('name'),
                away_team_data.get('id'): away_team_data.get('name')
            }
            teams_map = {k: v for k, v in teams_map.items() if k and v} # Clean out empty entries

            # --- Fetch Shotmap Data ---
            api_url = f"https://api.sofascore.com/api/v1/event/{event_id}/shotmap"
            api_data = page.evaluate(f"async (url) => {{ const response = await fetch(url); return await response.json(); }}", api_url)
            
            shots = pd.DataFrame(api_data.get('shotmap', []))
            if shots.empty:
                 print("API response was valid, but contained no shotmap data.")

            return {'shots': shots, 'teams': teams_map}

        except Exception as e:
            print(f"An error occurred with SofaScore scraping: {e}")
            return {'shots': pd.DataFrame(), 'teams': {}}
        finally:
            if not page.is_closed():
                browser.close()


async def scrape_fotmob_match(match_id: str):
    """
    Asynchronously scrapes all match data (general, stats, shotmap) from FotMob's API.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            api_url = f"https://www.fotmob.com/api/matchDetails?matchId={match_id}"
            await page.goto(api_url, timeout=60000)
            json_text = await page.locator('body').inner_text()
            data = json.loads(json_text)
            return data
        except Exception as e:
            print(f"Lỗi khi cào dữ liệu FotMob cho trận {match_id}: {e}")
            return None
        finally:
            await browser.close()

async def scrape_sofascore_match(fotmob_url: str):
    """
    Trích xuất dữ liệu shotmap. 
    Lưu ý: Để tăng tốc độ và độ tin cậy, hàm này cũng lấy dữ liệu từ API của FotMob,
    thay vì phải tìm và cào link của SofaScore.
    """
    match_id = get_match_id_from_url(fotmob_url)
    if not match_id:
        return {'shotmap': []}

    try:
        data = await scrape_fotmob_match(match_id)
        if data and 'content' in data and 'shotmap' in data['content']:
            # Trả về theo cấu trúc mà main_app mong đợi
            return {'shotmap': data['content']['shotmap']['shots']}
        else:
            return {'shotmap': []}
    except Exception as e:
        print(f"Lỗi khi lấy shotmap từ FotMob cho URL {fotmob_url}: {e}")
        return {'shotmap': []}


async def get_fotmob_team_recent_match_ids(team_name: str, num_matches: int = 3):
    """
    Tìm kiếm và trả về ID của N trận đấu đã hoàn thành gần đây nhất của một đội trên FotMob.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto("https://www.fotmob.com/", wait_until="domcontentloaded")

            search_input = page.get_by_placeholder("Search for team, player or league")
            await search_input.fill(team_name)
            
            # Chờ kết quả tìm kiếm và tìm liên kết chính xác của đội
            # Thường là liên kết đầu tiên có '/teams/' trong href và chứa đúng tên đội
            team_link_locator = page.locator(f'a[href*="/teams/"]:has-text("{team_name}")').first
            
            await team_link_locator.wait_for(state="visible", timeout=15000)
            team_url_path = await team_link_locator.get_attribute("href")
            
            if not team_url_path:
                raise ValueError(f"Không thể tìm thấy trang của đội '{team_name}'")

            team_page_url = f"https://www.fotmob.com{team_url_path}"
            await page.goto(team_page_url, wait_until="domcontentloaded")

            # Trích xuất dữ liệu JSON __NEXT_DATA__ từ mã nguồn trang
            next_data_script = await page.content()
            soup = BeautifulSoup(next_data_script, 'html.parser')
            next_data = soup.find('script', {'id': '__NEXT_DATA__'})
            
            if not next_data:
                raise ValueError("Không thể tìm thấy khối dữ liệu __NEXT_DATA__ trên trang của đội.")

            json_data = json.loads(next_data.string)
            
            fixtures = json_data.get('props', {}).get('pageProps', {}).get('fixtures', {}).get('allFixtures', {}).get('fixtures', [])
            
            if not fixtures:
                raise ValueError("Không tìm thấy danh sách trận đấu trong dữ liệu trang.")

            # Lọc các trận đã kết thúc và sắp xếp theo thời gian (mới nhất trước)
            completed_matches = [
                match for match in fixtures 
                if match.get('status', {}).get('finished') and not match.get('status', {}).get('cancelled')
            ]
            
            completed_matches.sort(key=lambda x: x.get('status', {}).get('utcTime', 0), reverse=True)

            match_ids = [str(match['id']) for match in completed_matches[:num_matches]]

            if len(match_ids) < num_matches:
                print(f"Cảnh báo: Chỉ tìm thấy {len(match_ids)} trận đã hoàn thành cho {team_name}, ít hơn {num_matches} trận yêu cầu.")

            if not match_ids:
                raise ValueError(f"Không có trận đấu nào đã hoàn thành được tìm thấy cho {team_name}.")

            return match_ids

        except PlaywrightTimeoutError:
            raise ValueError(f"Hết thời gian chờ khi tìm kiếm đội '{team_name}'. Tên đội có thể không chính xác hoặc không tồn tại trên FotMob.")
        except Exception as e:
            # Ném lại lỗi để lớp gọi có thể xử lý
            raise e
        finally:
            await browser.close()


def get_match_id_from_url(url: str) -> str:
    """Extracts match ID from a FotMob URL."""
    match = re.search(r'/matches/(\w+-\w+-\w+)/(\w+)', url)
    if match:
        return match.group(2)
    return None


if __name__ == '__main__':
    # Example Usage
    
    # 1. FotMob
    print("--- Scraping FotMob ---")
    fotmob_url = 'https://www.fotmob.com/matches/ac-milan-vs-roma/2gl9pd#4446402'
    fotmob_data = get_fotmob_match_data(fotmob_url)
    if not fotmob_data['shots_df'].empty:
        print("FotMob shotmap data retrieved successfully.")
        print(fotmob_data['shots_df'].head())
    print("\\n" + "="*30 + "\\n")

    # 2. Transfermarkt
    print("--- Scraping Transfermarkt ---")
    tm_url = "https://www.transfermarkt.us/erling-haaland/profil/spieler/418560"
    player_data = get_transfermarkt_player_data(tm_url)
    if player_data:
        print("Transfermarkt player data retrieved successfully.")
        # Print a subset of the data for brevity
        print(f"Name: {player_data.get('player_name')}")
        print(f"Agent: {player_data.get('agent')}")
        print(f"Contract Expiry: {player_data.get('contract_expiry')}")
    print("\\n" + "="*30 + "\\n")
    
    # 3. SofaScore
    print("--- Scraping SofaScore ---")
    sofascore_url = "https://www.sofascore.com/inter-miami-cf-new-york-red-bulls/gabsccKc#id:11911622"
    sofascore_data = get_sofascore_shotmap(sofascore_url)
    if not sofascore_data['shots'].empty:
        print(f"SofaScore shotmap data retrieved successfully. Found {len(sofascore_data['shots'])} shots.")
        # Print the first shot
        print(sofascore_data['shots'].iloc[0]) 