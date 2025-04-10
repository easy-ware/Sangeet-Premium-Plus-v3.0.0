from sangeet_premium.utils.util import send_email
MASTER_EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{subject}}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            padding: 20px;
        }
        .container {
            max-width: 500px;
            margin: auto;
            background-color: white; 
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .header {
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .header img {
            max-width: 200px;
            height: auto;
        }
        .code-block {
            background-color: #f0f8ff;
            padding: 10px;
            margin: 20px 0;
            font-size: 24px;
            letter-spacing: 5px;
            border: 2px dashed #add8e6;
        }
        .footer {
            border-top: 1px solid #ddd;
            padding-top: 10px;
            margin-top: 20px;
            text-align: center;
            font-size: 12px;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="https://your-app-url.com/logo.png" alt="Sangeet Premium Logo">
        </div>
        
        {{content}}
        
        <div class="footer">
            <p>Sent by Sangeet Premium &bull; <a href="https://your-app-url.com">Visit Our Website</a> &bull; <a href="https://your-app-url.com/unsubscribe">Unsubscribe</a></p>
        </div>
    </div>
</body>
</html>
"""
VERIFICATION_EMAIL_CONTENT = """
<h1>Your Verification Code</h1>

<p>Hi {{name}},</p>

<p>Here is your verification code:</p>

<div class="code-block">{{otp}}</div>

<p>Please enter this code in the app to complete your action.</p>

<p>If you didn't request this code, you can safely ignore this email.</p>
"""

PASSWORD_RESET_EMAIL_CONTENT = """
<h1>Reset Your Password</h1>

<p>Hi {{name}},</p>

<p>We received a request to reset your password. If you didn't make this request, you can safely ignore this email.</p>

<p>To reset your password, please click the button below:</p>

<a href="{{reset_link}}" style="display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; margin-top: 20px;">Reset Password</a>

<p>If the button doesn't work, you can also copy and paste this link into your browser:</p>

<p>{{reset_link}}</p>
"""

# Define other email content strings as needed
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Dashboard | Sangeet Premium</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #1a1037;
            --bg-secondary: #2d1854;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --accent-color: #ff3366;
            --input-bg: rgba(255, 255, 255, 0.1);
            --card-bg: rgba(45, 24, 84, 0.8);
        }

        body {
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .form-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .settings-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }

        .settings-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
        }

        .floating-note {
            animation: float 6s ease-in-out infinite;
        }

        @keyframes float {
            0% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(5deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }

        .btn-primary {
            background: var(--accent-color);
            transition: all 0.3s ease;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(255, 51, 102, 0.4);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }

        .qr-container {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
        }
    </style>
</head>
<body class="min-h-screen py-8 px-4">
    <!-- Floating Music Notes -->
    <div class="fixed inset-0 pointer-events-none z-0">
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 10%; left: 10%;">♪</div>
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 20%; left: 80%; animation-delay: 1s;">♫</div>
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 70%; left: 30%; animation-delay: 2s;">♩</div>
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 40%; left: 70%; animation-delay: 3s;">♪</div>
    </div>

    <div class="container max-w-4xl mx-auto relative z-10">
        <!-- Header -->
        <div class="text-center mb-8 animate__animated animate__fadeInDown">
            <h1 class="text-4xl font-bold mb-2">Welcome, {{ full_name }}! 🎵</h1>
            <p class="text-gray-400">Manage your account settings and preferences</p>
        </div>

        <!-- Main Content -->
        <div class="grid md:grid-cols-2 gap-6">
            <!-- Account Info Card -->
            <div class="form-card rounded-xl p-6 animate__animated animate__fadeInUp animate__delay-1s">
                <h2 class="text-xl font-semibold mb-4">Account Information</h2>
                
                <div class="space-y-4">
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Email</span>
                        <div class="flex items-center">
                            <span class="mr-2">{{ email }}</span>
                            
                        </div>
                    </div>

                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Username</span>
                        <span>{{ username }}</span>
                    </div>

                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Security Level</span>
                        <span class="px-2 py-1 text-sm rounded-full 
                            {% if twofa_method == 'both' %}
                                bg-purple-500 bg-opacity-20 text-purple-200
                            {% elif twofa_method in ['totp', 'email'] %}
                                bg-blue-500 bg-opacity-20 text-blue-200
                            {% else %}
                                bg-gray-500 bg-opacity-20 text-gray-200
                            {% endif %}
                        ">
                            {% if twofa_method == 'both' %}
                                🔒 Maximum Security
                            {% elif twofa_method == 'totp' %}
                                🔐 TOTP Enabled
                            {% elif twofa_method == 'email' %}
                                📧 Email 2FA
                            {% else %}
                                Basic Security
                            {% endif %}
                        </span>
                    </div>
                </div>
            </div>

            <!-- 2FA Management Card -->
            <div class="form-card rounded-xl p-6 animate__animated animate__fadeInUp animate__delay-2s">
                <h2 class="text-xl font-semibold mb-4">Two-Factor Authentication</h2>

                {% if error %}
                <div class="bg-red-500 bg-opacity-20 border border-red-400 text-red-200 px-4 py-3 rounded-lg mb-4">
                    {{ error }}
                </div>
                {% endif %}

                {% if success %}
                <div class="bg-green-500 bg-opacity-20 border border-green-400 text-green-200 px-4 py-3 rounded-lg mb-4">
                    {{ success }}
                </div>
                {% endif %}

                {% if setup_totp %}
                <div class="space-y-4 text-center">
                    <div class="mb-4">
                        <p class="text-sm text-gray-400 mb-4">Scan this QR code with your authenticator app</p>
                        <div class="qr-container inline-block mx-auto">
                            <img src="{{ qr_code }}" alt="TOTP QR Code" class="mx-auto">
                        </div>
                    </div>

                    <div class="text-sm text-gray-400 mb-4">
                        Or manually enter this code:<br>
                        <code class="bg-black bg-opacity-20 px-2 py-1 rounded text-sm">{{ totp_secret }}</code>
                    </div>

                    <form method="POST" action="{{ url_for('playback.verify_totp') }}" class="space-y-4">
                        <input type="hidden" name="setup_token" value="{{ setup_token }}">
                        <div>
                            <label class="block text-sm text-gray-400 mb-2">Enter the 6-digit code from your app</label>
                            <input type="text" name="code" 
                                   class="bg-white bg-opacity-10 border border-white border-opacity-20 rounded-lg px-4 py-2 w-full text-center text-xl tracking-widest"
                                   maxlength="6" pattern="\\d{6}" required>
                        </div>
                        <button type="submit" class="btn-primary w-full py-2 px-4 rounded-lg text-white">
                            Verify & Enable TOTP
                        </button>
                    </form>
                </div>
                {% else %}
                <div class="space-y-4">
                    {% if twofa_method == 'none' %}
                    <div class="space-y-3">
                        <form method="POST" action="{{ url_for('playback.enable_totp') }}">
                            <button type="submit" 
                                    class="w-full bg-green-500 bg-opacity-20 hover:bg-opacity-30 text-white py-3 px-4 rounded-lg transition-all transform hover:-translate-y-1">
                                🔐 Enable TOTP Authentication
                            </button>
                        </form>

                        <form method="POST" action="{{ url_for('playback.enable_email_2fa') }}">
                            <button type="submit" 
                                    class="w-full bg-blue-500 bg-opacity-20 hover:bg-opacity-30 text-white py-3 px-4 rounded-lg transition-all transform hover:-translate-y-1">
                                📧 Enable Email Authentication
                            </button>
                        </form>
                    </div>
                    {% elif twofa_method == 'totp' %}
                    <div class="text-center mb-4">
                        <div class="bg-green-500 bg-opacity-20 text-green-200 px-4 py-3 rounded-lg">
                            🔒 TOTP Authentication is Active
                        </div>
                    </div>
                    <form method="POST" action="{{ url_for('playback.enable_email_2fa') }}">
                        <button type="submit" 
                                class="w-full bg-blue-500 bg-opacity-20 hover:bg-opacity-30 text-white py-3 px-4 rounded-lg transition-all transform hover:-translate-y-1">
                            📧 Add Email Authentication
                        </button>
                    </form>
                    {% elif twofa_method == 'email' %}
                    <div class="text-center mb-4">
                        <div class="bg-blue-500 bg-opacity-20 text-blue-200 px-4 py-3 rounded-lg">
                            📧 Email Authentication is Active
                        </div>
                    </div>
                    <form method="POST" action="{{ url_for('playback.enable_totp') }}">
                        <button type="submit" 
                                class="w-full bg-green-500 bg-opacity-20 hover:bg-opacity-30 text-white py-3 px-4 rounded-lg transition-all transform hover:-translate-y-1">
                            🔐 Add TOTP Authentication
                        </button>
                    </form>
                    {% elif twofa_method == 'both' %}
                    <div class="text-center">
                        <div class="bg-purple-500 bg-opacity-20 text-purple-200 px-4 py-3 rounded-lg">
                            🔒 Maximum Security Enabled
                            <p class="text-sm mt-2">Both TOTP and Email authentication are active</p>
                        </div>
                    </div>
                    {% endif %}

                    {% if twofa_method != 'none' %}
                    <form method="POST" action="{{ url_for('playback.disable_2fa') }}" class="mt-4">
                        <button type="submit" onclick="return confirm('Are you sure you want to disable 2FA? This will reduce your account security.')"
                                class="w-full bg-red-500 bg-opacity-20 hover:bg-opacity-30 text-white py-3 px-4 rounded-lg transition-all transform hover:-translate-y-1">
                            ⚠ Disable Two-Factor Authentication
                        </button>
                    </form>
                    {% endif %}
                </div>
                {% endif %}
            </div>
        </div>

        <!-- Logout Section -->
        <div class="mt-8 text-center">
            <form method="POST" action="{{ url_for('playback.logout') }}">
                <button type="submit" class="btn-secondary px-8 py-3 rounded-lg text-white hover:text-white">
                    Sign Out
                </button>
            </form>
        </div>
    </div>

    {% if setup_totp %}
    <!-- Auto-focus OTP input when present -->
    <script>
        document.querySelector('input[name="code"]').focus();
    </script>
    {% endif %}
</body>
</html>
"""
LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Login | Sangeet Premium</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #1a1037;
            --bg-secondary: #2d1854;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --accent-color: #ff3366;
            --input-bg: rgba(255, 255, 255, 0.1);
            --card-bg: rgba(45, 24, 84, 0.8);
        }

        body {
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .form-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .custom-input {
            background: var(--input-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .custom-input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(255, 51, 102, 0.2);
        }

        .btn-primary {
            background: var(--accent-color);
            transition: all 0.3s ease;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(255, 51, 102, 0.4);
        }

        .floating-note {
            animation: float 6s ease-in-out infinite;
        }

        @keyframes float {
            0% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(5deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4">
    <!-- Floating Music Notes -->
    <div class="fixed inset-0 pointer-events-none z-0">
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 10%; left: 10%;">♪</div>
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 20%; left: 80%; animation-delay: 1s;">♫</div>
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 70%; left: 30%; animation-delay: 2s;">♩</div>
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 40%; left: 70%; animation-delay: 3s;">♪</div>
    </div>

    <div class="container max-w-md mx-auto z-10">
        <div class="text-center mb-8 animate__animated animate__fadeInDown">
            <h1 class="text-4xl font-bold mb-2">Sangeet Premium</h1>
            <p class="text-gray-300">Your Premium Music Experience</p>
        </div>

        <div class="form-card rounded-xl p-8 animate__animated animate__fadeInUp">
            {% if error %}
            <div class="bg-red-500 bg-opacity-20 border border-red-400 text-red-100 px-4 py-3 rounded-lg mb-6">
                {{ error }}
            </div>
            {% endif %}

            {% if success %}
            <div class="bg-green-500 bg-opacity-20 border border-green-400 text-green-100 px-4 py-3 rounded-lg mb-6">
                {{ success }}
            </div>
            {% endif %}

            {% if login_step == 'initial' %}
            <form method="POST" action="{{ url_for('playback.login') }}" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Email or Username</label>
                    <input type="text" name="login_id" class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none" required>
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Password</label>
                    <input type="password" name="password" class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none" required>
                </div>

                <button type="submit" class="btn-primary w-full py-3 px-4 rounded-lg text-white font-medium">
                    Continue
                </button>
            </form>

            {% elif login_step == '2fa' %}
            <form method="POST" action="{{ url_for('playback.login_verify') }}" class="space-y-6">
                <div class="text-center mb-6">
                    <div class="text-xl font-semibold mb-2">Two-Factor Authentication</div>
                    <p class="text-gray-300">Please enter the verification code to continue</p>
                </div>

                {% if twofa_method == 'email' %}
                <div class="text-sm text-gray-300 mb-4">
                    A verification code has been sent to your email
                </div>
                {% endif %}

                {% if twofa_method == 'totp' %}
                <div class="text-sm text-gray-300 mb-4">
                    Enter the code from your authenticator app
                </div>
                {% endif %}

                <div>
                    <input type="text" name="otp" class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none text-center text-2xl tracking-widest" maxlength="6" pattern="\\d{6}" required>
                </div>

                <input type="hidden" name="login_token" value="{{ login_token }}">
                
                <button type="submit" class="btn-primary w-full py-3 px-4 rounded-lg text-white font-medium">
                    Verify & Login
                </button>

                {% if twofa_method == 'email' %}
                <div class="text-center mt-4">
                    <button type="button" id="resendBtn" class="text-gray-400 hover:text-white text-sm" disabled>
                        Resend Code (180s)
                    </button>
                </div>

                <script>
                    let timeLeft = 180;
                    const resendBtn = document.getElementById('resendBtn');
                    
                    const countdown = setInterval(() => {
                        timeLeft--;
                        resendBtn.textContent = `Resend Code (${timeLeft}s)`;
                        
                        if (timeLeft <= 0) {
                            clearInterval(countdown);
                            resendBtn.textContent = 'Resend Code';
                            resendBtn.disabled = false;
                        }
                    }, 1000);

                    resendBtn.addEventListener('click', async () => {
                        if (!resendBtn.disabled) {
                            resendBtn.disabled = true;
                            timeLeft = 180;
                            
                            try {
                                const response = await fetch('/api/resend-otp', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json',
                                    },
                                    body: JSON.stringify({
                                        login_token: '{{ login_token }}'
                                    })
                                });
                                
                                if (response.ok) {
                                    const countdown = setInterval(() => {
                                        timeLeft--;
                                        resendBtn.textContent = `Resend Code (${timeLeft}s)`;
                                        
                                        if (timeLeft <= 0) {
                                            clearInterval(countdown);
                                            resendBtn.textContent = 'Resend Code';
                                            resendBtn.disabled = false;
                                        }
                                    }, 1000);
                                } else {
                                    resendBtn.textContent = 'Error. Try again';
                                    resendBtn.disabled = false;
                                }
                            } catch (error) {
                                resendBtn.textContent = 'Error. Try again';
                                resendBtn.disabled = false;
                            }
                        }
                    });
                    
                </script>
                {% endif %}
            </form>
            {% endif %}

            <div class="mt-6 flex flex-col space-y-3 text-center text-sm">
                <a href="{{ url_for('playback.register') }}" class="text-accent-color hover:text-opacity-80 transition-colors">
                    Create New Account
                </a>
                <a href="{{ url_for('playback.reset_password') }}" class="text-gray-400 hover:text-white transition-colors">
                    Reset Password
                </a>
                <a href="{{ url_for('playback.forgot_username') }}" class="text-gray-400 hover:text-white transition-colors">
                    Forgot Username?
                </a>
            </div>
        </div>
    </div>
    <script>
    async function setFavicons() {
                        const response = await fetch('/data/download/icons/login-system-login');
                        const data = await response.json();
                        const base64Data = data.base64;

                        const sizes = [16, 32, 48, 64, 128, 256];
                        sizes.forEach(size => {
                            const link = document.createElement('link');
                            link.rel = 'icon';
                            link.type = 'image/png';
                            link.sizes = `${size}x${size}`;
                            link.href = `data:image/png;base64,${base64Data}`;
                            document.head.appendChild(link);
                        });
                    }

                    // Call the function to set favicons
                    setFavicons();
    </script>
</body>
</html>
"""

REGISTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Register | Sangeet Premium</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #1a1037;
            --bg-secondary: #2d1854;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --accent-color: #ff3366;
            --input-bg: rgba(255, 255, 255, 0.1);
            --card-bg: rgba(45, 24, 84, 0.8);
        }

        body {
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .form-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .custom-input {
            background: var(--input-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .custom-input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(255, 51, 102, 0.2);
        }

        .btn-primary {
            background: var(--accent-color);
            transition: all 0.3s ease;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(255, 51, 102, 0.4);
        }

        .floating-note {
            animation: float 6s ease-in-out infinite;
        }

        @keyframes float {
            0% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(5deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }
        
        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 50;
            overflow-y: auto;
            background-color: rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease-out;
        }
        
        .modal-content {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            opacity: 0;
            transform: scale(0.9);
            transition: all 0.3s ease-out;
        }
        
        .modal.show {
            display: flex;
        }
        
        .modal.show .modal-content {
            opacity: 1;
            transform: scale(1);
        }
        
        .terms-content {
            max-height: 60vh;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.2);
            padding: 1rem;
            border-radius: 0.5rem;
        }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4">
    <!-- Floating Music Notes -->
    <div class="fixed inset-0 pointer-events-none z-0">
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 10%; left: 10%;">♪</div>
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 20%; left: 80%; animation-delay: 1s;">♫</div>
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 70%; left: 30%; animation-delay: 2s;">♩</div>
        <div class="floating-note absolute text-4xl text-white opacity-10" style="top: 40%; left: 70%; animation-delay: 3s;">♪</div>
    </div>

    <div class="container max-w-md mx-auto z-10">
        <div class="text-center mb-8 animate__animated animate__fadeInDown">
            <h1 class="text-4xl font-bold mb-2">Join Sangeet Premium</h1>
            <p class="text-gray-300">Create your account</p>
        </div>

        <div class="form-card rounded-xl p-8 animate__animated animate__fadeInUp">
            {% if error %}
            <div class="bg-red-500 bg-opacity-20 border border-red-400 text-red-100 px-4 py-3 rounded-lg mb-6">
                {{ error }}
            </div>
            {% endif %}

            {% if message %}
            <div class="bg-green-500 bg-opacity-20 border border-green-400 text-green-100 px-4 py-3 rounded-lg mb-6">
                {{ message }}
            </div>
            {% endif %}

            {% if register_step == 'verify' %}
            <div class="text-center mb-6">
                <div class="text-xl font-semibold mb-2">Verify Your Email</div>
                <p class="text-gray-300">Enter the verification code sent to {{ email }}</p>
            </div>

            <form method="POST" action="{{ url_for('playback.register_verify') }}" class="space-y-6">
                <div>
                    <input type="text" name="otp" 
                           class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none text-center text-2xl tracking-widest" 
                           maxlength="6" pattern="\\d{6}" required autofocus>
                </div>

                <input type="hidden" name="register_token" value="{{ register_token }}">
                
                <button type="submit" class="btn-primary w-full py-3 px-4 rounded-lg text-white font-medium">
                    Verify & Create Account
                </button>

                <div class="text-center mt-4">
                    <button type="button" id="resendBtn" class="text-gray-400 hover:text-white text-sm" disabled>
                        Resend Code (180s)
                    </button>
                </div>
            </form>

            <script>
                // Auto-focus OTP input
                document.querySelector('input[name="otp"]').focus();

                // Countdown timer for resend
                let timeLeft = 180;
                const resendBtn = document.getElementById('resendBtn');
                
                const countdown = setInterval(() => {
                    timeLeft--;
                    resendBtn.textContent = `Resend Code (${timeLeft}s)`;
                    
                    if (timeLeft <= 0) {
                        clearInterval(countdown);
                        resendBtn.textContent = 'Resend Code';
                        resendBtn.disabled = false;
                    }
                }, 1000);

                resendBtn.addEventListener('click', async () => {
                    if (!resendBtn.disabled) {
                        resendBtn.disabled = true;
                        timeLeft = 180;
                        
                        try {
                            const response = await fetch('/api/resend-otp', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({
                                    register_token: '{{ register_token }}'
                                })
                            });
                            
                            if (response.ok) {
                                const countdown = setInterval(() => {
                                    timeLeft--;
                                    resendBtn.textContent = `Resend Code (${timeLeft}s)`;
                                    
                                    if (timeLeft <= 0) {
                                        clearInterval(countdown);
                                        resendBtn.textContent = 'Resend Code';
                                        resendBtn.disabled = false;
                                    }
                                }, 1000);
                            } else {
                                resendBtn.textContent = 'Error. Try again';
                                resendBtn.disabled = false;
                            }
                        } catch (error) {
                            resendBtn.textContent = 'Error. Try again';
                            resendBtn.disabled = false;
                        }
                    }
                });
               
            </script>

            {% else %}
            <form method="POST" action="{{ url_for('playback.register') }}" class="space-y-6" id="registerForm">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Email</label>
                    <input type="email" name="email" class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none" required>
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Username</label>
                    <input type="text" name="username" class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none" required>
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Full Name</label>
                    <input type="text" name="full_name" class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none" required>
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Password</label>
                    <input type="password" name="password" class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none" required>
                </div>
                
                <div class="flex items-start mt-4">
                    <div class="flex items-center h-5">
                        <input id="terms" name="terms_accepted" type="checkbox" class="focus:ring-accent-color h-4 w-4 text-accent-color border-gray-300 rounded" required>
                    </div>
                    <div class="ml-3 text-sm">
                        <label for="terms" class="text-gray-300">I accept the <a href="#" id="termsLink" class="text-accent-color hover:underline">Terms and Conditions</a></label>
                    </div>
                </div>

                <button type="submit" class="btn-primary w-full py-3 px-4 rounded-lg text-white font-medium">
                    Create Account
                </button>
            </form>
            {% endif %}

            <div class="mt-6 text-center">
                <a href="{{ url_for('playback.login') }}" class="text-accent-color hover:text-opacity-80 transition-colors">
                    Back to Login
                </a>
            </div>
        </div>
    </div>
    
    <!-- Terms and Conditions Modal -->
    <div id="termsModal" class="modal">
        <div class="modal-content max-w-2xl mx-auto my-8 p-6 rounded-xl animate__animated animate__fadeIn">
            <div class="flex justify-between items-center mb-6">
                <h3 class="text-2xl font-bold">Terms and Conditions</h3>
                <button id="closeTerms" class="text-gray-400 hover:text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>
            
            <div class="terms-content" id="termsContent">
                <!-- Terms content will be loaded here -->
                <div class="animate-pulse">
                    <div class="h-4 bg-gray-700 rounded w-3/4 mb-4"></div>
                    <div class="h-4 bg-gray-700 rounded mb-4"></div>
                    <div class="h-4 bg-gray-700 rounded w-5/6 mb-4"></div>
                    <div class="h-4 bg-gray-700 rounded w-4/6 mb-4"></div>
                </div>
            </div>
            
            <div class="mt-6 flex justify-end">
                <button id="acceptTerms" class="btn-primary py-2 px-4 rounded-lg text-white font-medium mr-3">
                    Accept
                </button>
                <button id="declineTerms" class="bg-gray-600 hover:bg-gray-700 py-2 px-4 rounded-lg text-white font-medium">
                    Close
                </button>
            </div>
        </div>
    </div>
    
    <script>
        // Favicons
        async function setFavicons() {
            const response = await fetch('/data/download/icons/login-system-register');
            const data = await response.json();
            const base64Data = data.base64;

            const sizes = [16, 32, 48, 64, 128, 256];
            sizes.forEach(size => {
                const link = document.createElement('link');
                link.rel = 'icon';
                link.type = 'image/png';
                link.sizes = `${size}x${size}`;
                link.href = `data:image/png;base64,${base64Data}`;
                document.head.appendChild(link);
            });
        }

        // Call the function to set favicons
        setFavicons();
        
        // Terms and Conditions Modal Functionality
        const modal = document.getElementById('termsModal');
        const termsLink = document.getElementById('termsLink');
        const closeBtn = document.getElementById('closeTerms');
        const acceptBtn = document.getElementById('acceptTerms');
        const declineBtn = document.getElementById('declineTerms');
        const termsCheckbox = document.getElementById('terms');
        const termsContent = document.getElementById('termsContent');
        
        // Open modal when clicking the terms link
        termsLink.addEventListener('click', async (e) => {
            e.preventDefault();
            
            // Fetch terms content if not already loaded
            if (termsContent.innerHTML.includes('animate-pulse')) {
                try {
                    const response = await fetch('{{ url_for("playback.terms_register") }}');
                    if (response.ok) {
                        const data = await response.text();
                        termsContent.innerHTML = data;
                    } else {
                        termsContent.innerHTML = '<p class="text-red-400">Failed to load terms and conditions. Please try again later.</p>';
                    }
                } catch (error) {
                    termsContent.innerHTML = '<p class="text-red-400">Failed to load terms and conditions. Please try again later.</p>';
                }
            }
            
            modal.classList.add('show');
            
            // Add animation classes to modal content
            const modalContent = modal.querySelector('.modal-content');
            modalContent.classList.add('animate__animated', 'animate__zoomIn');
        });
        
        // Close modal functions
        const closeModal = () => {
            const modalContent = modal.querySelector('.modal-content');
            modalContent.classList.remove('animate__zoomIn');
            modalContent.classList.add('animate__zoomOut');
            
            setTimeout(() => {
                modal.classList.remove('show');
                modalContent.classList.remove('animate__zoomOut');
            }, 300);
        };
        
        closeBtn.addEventListener('click', closeModal);
        declineBtn.addEventListener('click', closeModal);
        
        // Accept terms functionality
        acceptBtn.addEventListener('click', () => {
            termsCheckbox.checked = true;
            closeModal();
        });
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
        
        // Form validation
        const registerForm = document.getElementById('registerForm');
        if (registerForm) {
            registerForm.addEventListener('submit', (e) => {
                if (!termsCheckbox.checked) {
                    e.preventDefault();
                    alert('You must accept the Terms and Conditions to create an account.');
                }
            });
        }
    </script>
</body>
</html>
"""

# Add these template variables to your app.py

FORGOT_USERNAME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Recover Username | Sangeet Premium</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #1a1037;
            --bg-secondary: #2d1854;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --accent-color: #ff3366;
            --input-bg: rgba(255, 255, 255, 0.1);
            --card-bg: rgba(45, 24, 84, 0.8);
        }

        body {
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .form-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .custom-input {
            background: var(--input-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .custom-input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(255, 51, 102, 0.2);
        }

        .btn-primary {
            background: var(--accent-color);
            transition: all 0.3s ease;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(255, 51, 102, 0.4);
        }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4">
    <div class="container max-w-md mx-auto z-10">
        <div class="text-center mb-8 animate__animated animate__fadeInDown">
            <h1 class="text-4xl font-bold mb-2">Recover Username</h1>
            <p class="text-gray-300">Enter your email to find your username</p>
        </div>

        <div class="form-card rounded-xl p-8 animate__animated animate__fadeInUp">
            {% if error %}
            <div class="bg-red-500 bg-opacity-20 border border-red-400 text-red-100 px-4 py-3 rounded-lg mb-6">
                {{ error }}
            </div>
            {% endif %}

            {% if step == 'email' %}
            <form method="POST" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Email Address</label>
                    <input type="email" name="email" class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none" required>
                </div>
                <button type="submit" class="btn-primary w-full py-3 px-4 rounded-lg text-white font-medium">
                    Send Recovery Code
                </button>
            </form>

            {% elif step == 'verify' %}
            <form method="POST" class="space-y-6">
                <p class="text-gray-300 mb-4">We sent a verification code to {{ email }}</p>
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Enter Code</label>
                    <input type="text" name="otp" maxlength="6" pattern="\d{6}"
                           class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none text-center text-2xl tracking-widest" 
                           required autofocus>
                </div>
                <input type="hidden" name="email_step" value="{{ email }}">
                <button type="submit" class="btn-primary w-full py-3 px-4 rounded-lg text-white font-medium">
                    Verify Code
                </button>
            </form>
            {% endif %}

            <div class="mt-6 text-center">
                <a href="{{ url_for('playback.login') }}" class="text-accent-color hover:text-opacity-80 transition-colors">
                    Back to Login
                </a>
            </div>
        </div>
    </div>
    <script>
       async function setFavicons() {
            const response = await fetch('/data/download/icons/login-system-forgot');
            const data = await response.json();
            const base64Data = data.base64;

            const sizes = [16, 32, 48, 64, 128, 256];
            sizes.forEach(size => {
                const link = document.createElement('link');
                link.rel = 'icon';
                link.type = 'image/png';
                link.sizes = `${size}x${size}`;
                link.href = `data:image/png;base64,${base64Data}`;
                document.head.appendChild(link);
            });
        }

        // Call the function to set favicons
        setFavicons();
    </script>
</body>
</html>
"""

RESET_PASSWORD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Reset Password | Sangeet Premium</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <style>
        /* Same styles as above */
        :root {
            --bg-primary: #1a1037;
            --bg-secondary: #2d1854;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --accent-color: #ff3366;
            --input-bg: rgba(255, 255, 255, 0.1);
            --card-bg: rgba(45, 24, 84, 0.8);
        }

        body {
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .form-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .custom-input {
            background: var(--input-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .custom-input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(255, 51, 102, 0.2);
        }

        .btn-primary {
            background: var(--accent-color);
            transition: all 0.3s ease;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(255, 51, 102, 0.4);
        }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4">
    <div class="container max-w-md mx-auto z-10">
        <div class="text-center mb-8 animate__animated animate__fadeInDown">
            <h1 class="text-4xl font-bold mb-2">Reset Password</h1>
            <p class="text-gray-300">Recover access to your account</p>
        </div>

        <div class="form-card rounded-xl p-8 animate__animated animate__fadeInUp">
            {% if error %}
            <div class="bg-red-500 bg-opacity-20 border border-red-400 text-red-100 px-4 py-3 rounded-lg mb-6">
                {{ error }}
            </div>
            {% endif %}

            {% if step == 'email' %}
            <form method="POST" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Email Address</label>
                    <input type="email" name="email" class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none" required>
                </div>
                <button type="submit" class="btn-primary w-full py-3 px-4 rounded-lg text-white font-medium">
                    Send Reset Code
                </button>
            </form>

            {% elif step == 'verify' %}
            <form method="POST" class="space-y-6">
                <p class="text-gray-300 mb-4">We sent a verification code to {{ email }}</p>
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Enter Code</label>
                    <input type="text" name="otp" maxlength="6" pattern="\d{6}"
                           class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none text-center text-2xl tracking-widest" 
                           required autofocus>
                </div>
                <input type="hidden" name="reset_email" value="{{ email }}">
                <button type="submit" class="btn-primary w-full py-3 px-4 rounded-lg text-white font-medium">
                    Verify Code
                </button>
            </form>

            {% elif step == 'new_password' %}
            <form method="POST" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">New Password</label>
                    <input type="password" name="new_password" 
                           class="custom-input w-full px-4 py-3 rounded-lg focus:outline-none"
                           required minlength="6">
                </div>
                <input type="hidden" name="final_user_id" value="{{ user_id }}">
                <button type="submit" class="btn-primary w-full py-3 px-4 rounded-lg text-white font-medium">
                    Update Password
                </button>
            </form>
            {% endif %}

            <div class="mt-6 text-center">
                <a href="{{ url_for('playback.login') }}" class="text-accent-color hover:text-opacity-80 transition-colors">
                    Back to Login
                </a>
            </div>
        </div>
    </div>
    <script>
        async function setFavicons() {
        const response = await fetch('/data/download/icons/login-system-forgot');
        const data = await response.json();
        const base64Data = data.base64;

        const sizes = [16, 32, 48, 64, 128, 256];
        sizes.forEach(size => {
            const link = document.createElement('link');
            link.rel = 'icon';
            link.type = 'image/png';
            link.sizes = `${size}x${size}`;
            link.href = `data:image/png;base64,${base64Data}`;
            document.head.appendChild(link);
        });
    }

    // Call the function to set favicons
    setFavicons();
    </script>
</body>
</html>
"""





# Email template variations for different scenarios
def get_base_template(content_section):
    """Base template that wraps the content section"""
    return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sangeet Premium</title>
</head>
<body style="margin: 0; padding: 0; font-family: Arial, sans-serif; line-height: 1.6; background-color: #f0f2f5;">
    <style>
        @media (prefers-color-scheme: dark) {{
            .email-body {{ background-color: #1a1a1a !important; }}
            .email-container {{ background-color: #2d2d2d !important; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important; }}
            .header {{ background: linear-gradient(135deg, #8f6cf5, #6b3ce7) !important; }}
            .content-text {{ color: #ffffff !important; }}
            .secondary-text {{ color: #bbbbbb !important; }}
            .feature-box {{ background-color: #363636 !important; }}
            .system-notice {{ background-color: #363636 !important; border-color: #444444 !important; }}
            .code-box {{ background-color: #363636 !important; }}
        }}
    </style>

    <table class="email-body" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color: #f0f2f5; padding: 20px;">
        <tr>
            <td align="center">
                <table class="email-container" width="600" cellpadding="0" cellspacing="0" border="0" style="background-color: #ffffff; border-radius: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); max-width: 600px;">
                    <!-- Header -->
                    <tr>
                        <td class="header" style="background: linear-gradient(135deg, #6b3ce7, #4c2ba8); padding: 30px; border-radius: 20px 20px 0 0; text-align: center;">
                            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                                <tr>
                                    <td align="center">
                                        <span style="font-size: 32px; vertical-align: middle;">🎵</span>
                                        <span style="font-size: 28px; font-weight: bold; margin-left: 10px; color: #ffffff;">Sangeet Premium</span>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- System Notice -->
                    <tr>
                        <td style="padding: 20px 40px 0;">
                            <table width="100%" cellpadding="0" cellspacing="0" border="0" class="system-notice" style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;">
                                <tr>
                                    <td style="padding: 15px;">
                                        <p style="margin: 0; color: #666666; font-size: 13px; text-align: center;">
                                            ⚡ This is an automated system-generated email. Please do not reply to this message.
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Content Section -->
                    {content_section}

                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f8f9fa; padding: 25px; border-radius: 0 0 20px 20px; text-align: center;">
                            <p class="secondary-text" style="margin: 0; color: #666666; font-size: 12px;">© 2025 Sangeet Premium. All rights reserved.</p>
                            <p class="secondary-text" style="margin: 10px 0 0 0; color: #666666; font-size: 12px;">You received this email because you're subscribed to Sangeet Premium updates.</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
'''

# Template for forgot password emails
FORGOT_PASSWORD_TEMPLATE = '''
    <!-- Content -->
    <tr>
        <td style="padding: 40px;">
            <div class="content-text" style="color: #333333;">
                <h1 style="margin: 0 0 20px 0; font-size: 24px; font-weight: 600;">Reset Your Password</h1>
                
                <div style="background-color: #fff4e5; border-radius: 12px; padding: 15px; margin-bottom: 25px;">
                    <p style="margin: 0; color: #b25f00; font-size: 14px;">
                        🔒 We received a request to reset your password. If you didn't make this request, please ignore this email.
                    </p>
                </div>

                <p style="margin: 0 0 25px 0;">Your password reset code is:</p>
                
                <div class="code-box" style="background-color: #f8f9fa; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 25px;">
                    <span style="font-size: 32px; font-weight: bold; letter-spacing: 3px; color: #6b3ce7;">{code}</span>
                </div>

                <p style="margin: 0 0 10px 0; font-size: 14px;">This code will expire in 10 minutes.</p>
                <p style="margin: 0; font-size: 14px; color: #666666;">For security reasons, please do not share this code with anyone.</p>
            </div>
        </td>
    </tr>
'''

# Template for registration OTP emails
REGISTER_OTP_TEMPLATE = '''
    <!-- Content -->
    <tr>
        <td style="padding: 40px;">
            <div class="content-text" style="color: #333333;">
                <h1 style="margin: 0 0 20px 0; font-size: 24px; font-weight: 600;">Welcome to Sangeet Premium! 🎉</h1>
                
                <p style="margin: 0 0 25px 0;">Thank you for joining us! To complete your registration, please use this verification code:</p>
                
                <div class="code-box" style="background-color: #f8f9fa; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 25px;">
                    <span style="font-size: 32px; font-weight: bold; letter-spacing: 3px; color: #6b3ce7;">{code}</span>
                </div>

                <div style="background-color: #e8f5e9; border-radius: 12px; padding: 20px; margin-bottom: 25px;">
                    <h3 style="margin: 0 0 10px 0; color: #2e7d32; font-size: 16px;">What's Next?</h3>
                    <ul style="margin: 0; padding-left: 20px; color: #2e7d32;">
                        <li>Enter this code to verify your email</li>
                        <li>Complete your profile setup</li>
                        <li>Start enjoying premium music!</li>
                    </ul>
                </div>

                <p style="margin: 0; font-size: 14px; color: #666666;">This code will expire in 10 minutes.</p>
            </div>
        </td>
    </tr>
'''

# Template for password reset confirmation emails
RESET_CONFIRMATION_TEMPLATE = '''
    <!-- Content -->
    <tr>
        <td style="padding: 40px;">
            <div class="content-text" style="color: #333333;">
                <h1 style="margin: 0 0 20px 0; font-size: 24px; font-weight: 600;">Password Successfully Reset</h1>
                
                <div style="background-color: #e8f5e9; border-radius: 12px; padding: 15px; margin-bottom: 25px;">
                    <p style="margin: 0; color: #2e7d32;">
                        ✅ Your password has been successfully changed.
                    </p>
                </div>

                <div style="background-color: #f8f9fa; border-radius: 12px; padding: 20px; margin-bottom: 25px;">
                    <h3 style="margin: 0 0 15px 0; font-size: 16px;">Important Security Information:</h3>
                    <ul style="margin: 0; padding-left: 20px; color: #666666;">
                        <li style="margin-bottom: 10px">This change was made on {date} at {time}</li>
                        <li style="margin-bottom: 10px">Location: {location}</li>
                        <li>Device: {device}</li>
                    </ul>
                </div>

                <p style="margin: 0 0 15px 0;">If you didn't make this change, please contact our support team immediately.</p>
                
                <div style="background-color: #fff4e5; border-radius: 12px; padding: 15px; margin-bottom: 25px;">
                    <p style="margin: 0; color: #b25f00; font-size: 14px;">
                        🔔 For security, you'll need to re-login on all your devices.
                    </p>
                </div>
            </div>
        </td>
    </tr>
'''

def send_forgot_password_email(to_email, code):
    """Send forgot password email"""
    content = FORGOT_PASSWORD_TEMPLATE.format(code=code)
    full_template = get_base_template(content)
    return send_email(to_email, "Reset Your Password - Sangeet Premium", full_template)

def send_register_otp_email(to_email, code):
    """Send registration OTP email"""
    content = REGISTER_OTP_TEMPLATE.format(code=code)
    full_template = get_base_template(content)
    return send_email(to_email, "Welcome to Sangeet Premium - Verify Your Email", full_template)

def send_reset_confirmation_email(to_email, date, time, location, device):
    """Send password reset confirmation email"""
    content = RESET_CONFIRMATION_TEMPLATE.format(
        date=date,
        time=time,
        location=location,
        device=device
    )
    full_template = get_base_template(content)
    return send_email(to_email, "Password Reset Successful - Sangeet Premium", full_template)


FORGOT_USERNAME_TEMPLATE = '''
    <!-- Content -->
    <tr>
        <td style="padding: 40px;">
            <div class="content-text" style="color: #333333;">
                <h1 style="margin: 0 0 20px 0; font-size: 24px; font-weight: 600;">Your Username Recovery</h1>
                
                <div style="background-color: #e3f2fd; border-radius: 12px; padding: 15px; margin-bottom: 25px;">
                    <p style="margin: 0; color: #0d47a1; font-size: 14px;">
                        👋 We received a request to recover your username for Sangeet Premium.
                    </p>
                </div>

                <p style="margin: 0 0 20px 0;">Here are your account details:</p>
                
                <div class="code-box" style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px; padding: 20px; margin-bottom: 25px;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0">
                        <tr>
                            <td style="padding-bottom: 15px;">
                                <p style="margin: 0; font-size: 14px; color: #666666;">Your username is:</p>
                                <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #6b3ce7;">{username}</p>
                            </td>
                        </tr>
                        <tr>
                            <td>
                                <p style="margin: 0; font-size: 14px; color: #666666;">Associated email:</p>
                                <p style="margin: 5px 0 0 0; font-size: 16px; color: #333333;">{email}</p>
                            </td>
                        </tr>
                    </table>
                </div>

                <div style="background-color: #fafafa; border-radius: 12px; padding: 20px; margin-bottom: 25px;">
                    <h3 style="margin: 0 0 15px 0; font-size: 16px;">Quick Tips:</h3>
                    <ul style="margin: 0; padding-left: 20px; color: #666666;">
                        <li style="margin-bottom: 10px">Save your username in a secure location</li>
                        <li style="margin-bottom: 10px">Consider using a password manager</li>
                        <li>Enable two-factor authentication for extra security</li>
                    </ul>
                </div>

                <div style="background-color: #fff4e5; border-radius: 12px; padding: 15px; margin-bottom: 25px;">
                    <p style="margin: 0; color: #b25f00; font-size: 14px;">
                        🔒 If you didn't request this username recovery, please contact our support team immediately.
                    </p>
                </div>

                <p style="margin: 0; font-size: 14px; color: #666666;">Need help? Visit our support center for assistance.</p>
            </div>
        </td>
    </tr>
'''

def send_forgot_username_email(to_email, username):
    """Send forgot username email"""
    content = FORGOT_USERNAME_TEMPLATE.format(
        username=username,
        email=to_email
    )
    full_template = get_base_template(content)
    return send_email(to_email, "Your Username Recovery - Sangeet Premium", full_template)


