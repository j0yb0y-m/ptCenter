# ptCenter - Advanced Pentesting Assistant 🛡️

**ptCenter** is a professional-grade CLI penetration testing framework developed in Python. It streamlines the reconnaissance and scanning phases of a security audit by integrating industry-standard tools with AI-driven vulnerability analysis via Google Gemini.

Designed for security researchers and ethical hackers, ptCenter automates the workflow from raw data collection to intelligent risk assessment.

## 🚀 Key Features

* **Intelligent Network Scanning:** Seamless integration with `Nmap` for service discovery and port auditing.
* **AI-Powered Analysis:** Leverages Google's **Gemini Pro** to interpret scan results, identify potential CVEs, and suggest remediation steps.
* **Comprehensive Reconnaissance:** Automated subdomain enumeration using a multi-tool approach (Sublist3r, Amass, Subfinder).
* **Directory Discovery:** Built-in web path brute-forcing using `dirb`.
* **Professional Logging:** Automatic session logging and timestamped output management for audit trails.
* **Modular Architecture:** Clean, object-oriented Python code designed for extensibility.

## 🛠️ Prerequisites

Before running ptCenter, ensure you have the following tools installed on your Linux system:

* **Python 3.8+**
* **Nmap**
* **Dirb**
* **Sublist3r / Amass / Subfinder** (for full recon capabilities)

## 📦 Installation

1. **Clone the repository:**
```bash
   git clone [https://github.com/yourusername/ptCenter.git](https://github.com/j0yb0y-m/ptCenter.git)
   
   cd ptCenter

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Configure Environment Variables:**
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here

```



## 💻 Usage

Run the tool with root privileges to ensure all scanning features work correctly:

```bash
sudo python3 ptCenter.py

```

### Main Menu Options:

1. **Nmap Scan:** Perform network discovery and get AI-based vulnerability insights.
2. **Directory Brute Force:** Identify hidden paths on web servers.
3. **Subdomain Scan:** Comprehensive external attack surface mapping.
4. **AI Analysis:** Manual analysis of existing scan logs.

## 📂 Output Structure

All scan results and AI reports are automatically saved to:
`tmp/ptcenter_outputs/`

## ⚠️ Disclaimer

This tool is intended for **educational purposes and authorized penetration testing only**. Unauthorized access to computer systems is illegal. The developer (@j0yb0y-m) is not responsible for any misuse of this software. Always obtain written consent before testing any target.

---

**Developed by:** [Mahdi (@j0yb0y-m)](https://github.com/j0yb0y-m)

*Cyber Security Engineering Student | Junior Pentester | Programmer*