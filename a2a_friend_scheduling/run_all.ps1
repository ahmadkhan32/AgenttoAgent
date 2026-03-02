Write-Host "Starting Kaitlynn Agent..."
Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "cd kaitlynn_agent_langgraph; uv venv; .\.venv\Scripts\activate.ps1; uv run --active app\__main__.py"

Write-Host "Starting Nate Agent..."
Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "cd nate_agent_crewai; uv venv; .\.venv\Scripts\activate.ps1; uv run --active ."

Write-Host "Starting Karley Agent..."
Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "cd karley_agent_adk; uv venv; .\.venv\Scripts\activate.ps1; uv run --active ."

Write-Host "Starting Host Agent..."
Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "cd host_agent_adk; uv venv; .\.venv\Scripts\activate.ps1; uv run --active adk web"

Write-Host "All agents started in separate windows."
