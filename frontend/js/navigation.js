/**
 * Global Navigation Bar for Physical AI Platform
 * Add this script to any page to include the top navigation
 */

(function() {
    const navigationHTML = `
        <style>
            .platform-nav {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                backdrop-filter: blur(10px);
                padding: 12px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                z-index: 9999;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            }

            .platform-nav * {
                box-sizing: border-box;
            }

            .nav-brand {
                display: flex;
                align-items: center;
                gap: 10px;
                color: white;
                text-decoration: none;
                font-weight: 700;
                font-size: 18px;
            }

            .nav-brand:hover {
                opacity: 0.9;
            }

            .nav-links {
                display: flex;
                gap: 5px;
                align-items: center;
                flex-wrap: wrap;
            }

            .nav-link {
                color: white;
                text-decoration: none;
                padding: 8px 15px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                transition: all 0.3s;
                background: rgba(255, 255, 255, 0.1);
                white-space: nowrap;
            }

            .nav-link:hover {
                background: rgba(255, 255, 255, 0.25);
                transform: translateY(-2px);
            }

            .nav-link.active {
                background: rgba(255, 255, 255, 0.3);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }

            .nav-dropdown {
                position: relative;
            }

            .nav-dropdown-content {
                display: none;
                position: absolute;
                top: 100%;
                left: 0;
                background: white;
                min-width: 200px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
                border-radius: 8px;
                margin-top: 5px;
                overflow: hidden;
            }

            .nav-dropdown:hover .nav-dropdown-content {
                display: block;
            }

            .nav-dropdown-link {
                display: block;
                padding: 12px 16px;
                color: #333;
                text-decoration: none;
                font-size: 13px;
                font-weight: 500;
                transition: background 0.2s;
            }

            .nav-dropdown-link:hover {
                background: #f3f4f6;
            }

            .nav-spacer {
                height: 52px;
            }

            .mobile-menu-btn {
                display: none;
                background: rgba(255, 255, 255, 0.2);
                border: none;
                color: white;
                padding: 8px 12px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 18px;
            }

            @media (max-width: 968px) {
                .nav-links {
                    display: none;
                    position: absolute;
                    top: 100%;
                    left: 0;
                    right: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    flex-direction: column;
                    padding: 10px;
                    gap: 5px;
                }

                .nav-links.mobile-open {
                    display: flex;
                }

                .mobile-menu-btn {
                    display: block;
                }

                .nav-dropdown-content {
                    position: static;
                    display: none;
                    background: rgba(255, 255, 255, 0.1);
                    margin-top: 5px;
                    box-shadow: none;
                }

                .nav-dropdown.mobile-open .nav-dropdown-content {
                    display: block;
                }

                .nav-dropdown-link {
                    color: white;
                }

                .nav-link {
                    width: 100%;
                    text-align: left;
                }
            }
        </style>

        <nav class="platform-nav">
            <a href="/" class="nav-brand">
                🤖 Physical AI Platform
            </a>

            <button class="mobile-menu-btn" onclick="toggleMobileMenu()">☰</button>

            <div class="nav-links" id="navLinks">
                <a href="/" class="nav-link">🏠 Home</a>
                
                <div class="nav-dropdown">
                    <a href="#" class="nav-link" onclick="toggleDropdown(event)">🎬 Video ▾</a>
                    <div class="nav-dropdown-content">
                        <a href="/frontend/video-streaming" class="nav-dropdown-link">📹 Video Streaming</a>
                        <a href="/frontend/video-streaming#tutorials" class="nav-dropdown-link">📚 Tutorials</a>
                        <a href="/frontend/video-streaming#guided" class="nav-dropdown-link">🎓 Guided Ops</a>
                        <a href="/frontend/video-streaming#recordings" class="nav-dropdown-link">🎬 Recordings</a>
                    </div>
                </div>

                <div class="nav-dropdown">
                    <a href="#" class="nav-link" onclick="toggleDropdown(event)">🥽 AR/VR ▾</a>
                    <div class="nav-dropdown-content">
                        <a href="/frontend/ar-vr-viewer" class="nav-dropdown-link">🥽 AR/VR Viewer</a>
                        <a href="/frontend/ar-vr-viewer#environments" class="nav-dropdown-link">🌍 Environments</a>
                    </div>
                </div>

                <div class="nav-dropdown">
                    <a href="#" class="nav-link" onclick="toggleDropdown(event)">🤖 Robot ▾</a>
                    <div class="nav-dropdown-content">
                        <a href="/frontend/robot-3d-viewer" class="nav-dropdown-link">🤖 3D Viewer</a>
                        <a href="/dashboard" class="nav-dropdown-link">📊 Dashboard</a>
                        <a href="/frontend/robot-details" class="nav-dropdown-link">🔍 Details</a>
                    </div>
                </div>

                <a href="/frontend/ai-support-assistant" class="nav-link">💬 AI Assistant</a>
                <a href="/frontend/robot-alerts-pro-dashboard" class="nav-link">🚨 Alerts</a>
                <a href="/frontend/damage-assessment" class="nav-link">📸 Assessment</a>
                <a href="/api/docs" class="nav-link">📖 API Docs</a>
            </div>
        </nav>

        <div class="nav-spacer"></div>
    `;

    // Insert navigation at the beginning of body
    if (document.body) {
        document.body.insertAdjacentHTML('afterbegin', navigationHTML);
        
        // Highlight active page
        const currentPath = window.location.pathname;
        document.querySelectorAll('.nav-link, .nav-dropdown-link').forEach(link => {
            const linkPath = new URL(link.href, window.location.origin).pathname;
            if (linkPath === currentPath || (linkPath !== '/' && currentPath.includes(linkPath))) {
                link.classList.add('active');
            }
        });
    } else {
        // If body is not ready, wait for DOMContentLoaded
        document.addEventListener('DOMContentLoaded', () => {
            document.body.insertAdjacentHTML('afterbegin', navigationHTML);
        });
    }

    // Add global functions for mobile menu
    window.toggleMobileMenu = function() {
        document.getElementById('navLinks').classList.toggle('mobile-open');
    };

    window.toggleDropdown = function(event) {
        event.preventDefault();
        if (window.innerWidth <= 968) {
            event.target.parentElement.classList.toggle('mobile-open');
        }
    };
})();
