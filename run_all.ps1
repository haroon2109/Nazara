$env:MOCK_NAZARA="1"
Start-Process -NoNewWindow python -ArgumentList "app.py"
Start-Sleep -Seconds 5
python verify_app.py
