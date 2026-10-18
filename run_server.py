import uvicorn
import webbrowser
import time
import threading

def open_browser():
    time.sleep(1.5)
    print("\n[+] Opening web browser at http://127.0.0.1:8000 ...")
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

if __name__ == "__main__":
    print("=" * 65)
    print(" AI MODEL DRIFT DETECTIVE - MLOps Monitoring Dashboard")
    print(" Monitors deployed customer-churn classifier for silent degradation")
    print("=" * 65)
    print("[*] Starting backend server on http://127.0.0.1:8000")
    print("[*] Demo Logins:")
    print("    - Admin:    admin / admin123")
    print("    - Analyst:  analyst / analyst123")
    print("=" * 65)
    
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
