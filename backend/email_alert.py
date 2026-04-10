import smtplib
from email.mime.text import MIMEText
from datetime import datetime

def send_email_alert(unsafe_parameters, station_code="WQ-STATION-01", receivers=None):
    """
    Send email alert with unsafe water parameters to one or multiple receivers
    
    Args:
        unsafe_parameters: List of dicts with keys: 'parameter', 'value', 'threshold_type', 'threshold_value'
                          Can be empty list if water is unsafe but no individual parameters violate thresholds
        station_code: Water monitoring station identifier
        receivers: Single email (str) or list of emails (list). Default: ["achsagracin@gmail.com"]
    """
    sender = "lymphaaalerts@gmail.com"
    app_password = "dbol bpkp caih xbne"
    
    # Handle receivers parameter - convert to list if necessary
    if receivers is None:
        receivers = ["achsagracin@gmail.com"]
    elif isinstance(receivers, str):
        receivers = [receivers]
    elif not isinstance(receivers, list):
        receivers = list(receivers)
    
    # Validate receivers
    if not receivers:
        print("[ERROR] No receivers provided for email alert")
        return False

    # Validate input - allow empty parameters list (model may detect unsafe pattern)
    print(f"[EMAIL_ALERT] Called with {len(unsafe_parameters)} parameters for station {station_code}")
    print(f"[EMAIL_ALERT] Sending to {len(receivers)} recipient(s): {', '.join(receivers)}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build parameter details HTML
    params_html = ""
    if unsafe_parameters:
        for param in unsafe_parameters:
            try:
                params_html += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>{param['parameter']}</strong></td>
            <td style="padding: 10px; border-bottom: 1px solid #eee;">{float(param['value']):.2f}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee;">
                <span style="color: #ff0000; font-weight: bold;">{param['threshold_type'].upper()}</span> {param['threshold_value']}
            </td>
        </tr>
        """
            except Exception as e:
                print(f"[ERROR] Failed to format parameter: {param} - {e}")
                continue
    else:
        # No specific parameters violated thresholds, but model detected unsafe condition
        params_html = """
        <tr>
            <td colspan="3" style="padding: 15px; text-align: center; color: #666;">
                <em>Water Quality Model detected unsafe conditions. No individual parameters exceed safety thresholds, but the integrated analysis indicates unsafe water quality.</em>
            </td>
        </tr>
        """
    
    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
            <div style="background-color: white; padding: 25px; border-radius: 10px; border-left: 5px solid #ff0000; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                
                <h2 style="color: #ff0000; margin: 0 0 20px 0;">🚨 WATER QUALITY ALERT 🚨</h2>
                
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #ff0000;">
                    <h3 style="color: #ff0000; margin: 0 0 10px 0;">⚠️ UNSAFE WATER DETECTED</h3>
                    <p style="margin: 0; font-size: 16px; color: #333;">
                        <strong>Status:</strong> <span style="color: #ff0000;">UNSAFE</span><br>
                        <strong>Station Code:</strong> <span style="font-family: monospace; background-color: #f0f0f0; padding: 3px 8px; border-radius: 3px;">{station_code}</span><br>
                        <strong>Alert Time:</strong> {timestamp}
                    </p>
                </div>
                
                <hr style="border: none; border-top: 2px solid #ddd; margin: 20px 0;">
                
                <h3 style="color: #333; margin-bottom: 15px;">❌ Water Quality Analysis:</h3>
                
                <table style="width: 100%; border-collapse: collapse; background-color: #f9f9f9;">
                    <thead>
                        <tr style="background-color: #ff6b6b; color: white;">
                            <th style="padding: 12px; text-align: left;">Parameter</th>
                            <th style="padding: 12px; text-align: left;">Current Value</th>
                            <th style="padding: 12px; text-align: left;">Threshold Violation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {params_html}
                    </tbody>
                </table>
                
                <hr style="border: none; border-top: 2px solid #ddd; margin: 20px 0;">
                
                <div style="background-color: #ffe6e6; padding: 15px; border-radius: 5px; margin: 15px 0;">
                    <h3 style="color: #ff0000; margin: 0 0 10px 0;">⚡ IMMEDIATE ACTION REQUIRED:</h3>
                    <ul style="margin: 0; padding-left: 20px; color: #333;">
                        <li>Check your water source immediately</li>
                        <li>Do not use this water for drinking or cooking</li>
                        <li>Verify all parameters at the monitoring station</li>
                        <li>Take corrective action as needed</li>
                    </ul>
                </div>
                
                <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                
                <p style="font-size: 12px; color: #999; margin: 0;">
                    <strong>System:</strong> LYMPHA Water Quality Monitoring Dashboard<br>
                    <strong>Priority:</strong> URGENT<br>
                    <em>This is an automated alert. Please verify readings and take appropriate action.</em>
                </p>
            </div>
        </body>
    </html>
    """

    try:
        print(f"[EMAIL_ALERT] Connecting to SMTP ({station_code})...")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            print(f"[EMAIL_ALERT] Connected. Starting TLS...")
            server.starttls()
            print(f"[EMAIL_ALERT] TLS started. Logging in...")
            server.login(sender, app_password)
            print(f"[EMAIL_ALERT] Login successful. Creating message...")
            
            # Send to each receiver
            for receiver in receivers:
                msg = MIMEText(html_body, 'html')
                msg["Subject"] = f"🚨 LYMPHA ALERT: Unsafe Water at {station_code}"
                msg["From"] = sender
                msg["To"] = receiver
                
                print(f"[EMAIL_ALERT] Sending message to {receiver}...")
                server.send_message(msg)
                print(f"[EMAIL_ALERT] ✅ SUCCESS: Email sent to {receiver}")
        
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"[EMAIL_ERROR] Authentication failed: {e}")
        print(f"[EMAIL_ERROR] Check: Sender={sender}, App password valid?")
        return False
    except smtplib.SMTPException as e:
        print(f"[EMAIL_ERROR] SMTP Error: {e}")
        return False
    except Exception as e:
        print(f"[EMAIL_ERROR] Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
