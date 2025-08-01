// Global state
let currentUser = null;
let selectedMood = null;
let greetingShown = false;
let voiceMode = false;
let recognition = null;


// Show page
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(pageId).classList.add('active');
}

function showLogin(role) {
    showPage(role === 'student' ? 'student-login' : 'teacher-login');
}

// Student Signup
function handleStudentSignup(event) {
    event.preventDefault();
    const name = document.getElementById('signup-username').value.trim();
    const email = document.getElementById('signup-email').value.trim();
    const password = document.getElementById('signup-password').value.trim();
    clearMessages();
    if (!name || !email || !password) return showMessage('Please fill in all fields.', 'error');
    if (password.length < 6) return showMessage('Password must be at least 6 characters long.', 'error');

    fetch('http://localhost:5000/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            showMessage('Account created! Please sign in.', 'success');
            document.getElementById('student-signup-form').reset();
            setTimeout(() => showPage('student-login'), 2000);
        } else showMessage(data.message, 'error');
    })
    .catch(() => showMessage('Server error. Try again later.', 'error'));
}

// Student Login
function handleStudentLogin(event) {
    event.preventDefault();

    const loginBtn = document.getElementById('student-login-btn');
    loginBtn.disabled = true; 

    const email = document.getElementById('student-username').value.trim();
    const password = document.getElementById('student-password').value.trim();
    clearMessages();
    if (!email || !password){ 
        showMessage('Please enter both email and password.', 'error');
        loginBtn.disabled = false;
        return;
    }
    fetch('http://localhost:5000/student-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    })
    .then(res => res.json())
    .then(data => {
        loginBtn.disabled = false; // Re-enable button after response
        if (data.user) {
            currentUser = data.user;
            selectedMood = null;
            greetingShown = false;
            showPage('student-dashboard');
            loadChatHistory();
            if(data.user.greeting)
            {
                addAIResponse(data.user.greeting)
                greetingShown = true;
            }
        } 
        else{ 
        showMessage(data.error, 'error');}
    })
    .catch(() => showMessage('Server error. Try again later.', 'error'));
}

// Teacher Login - UPDATED WITH DASHBOARD LOADING
function handleTeacherLogin(event) {
    event.preventDefault();
    const teacherId = document.getElementById('teacher-id').value.trim();
    const password = document.getElementById('teacher-password').value.trim();
    clearMessages();
    if (!teacherId || !password) return showMessage('Please enter both Teacher ID and password.', 'error');

    fetch('http://localhost:5000/teacher-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ teacherId, password })
    })
    .then(res => res.json())
    .then(data => {
        if (data.user) {
            currentUser = data.user;
            showPage('teacher-dashboard');
            // Load teacher dashboard data
            loadTeacherDashboard();
        } else showMessage(data.error, 'error');
    })
    .catch(() => showMessage('Server error. Try again later.', 'error'));
}

// NEW: Load Teacher Dashboard Data
function loadTeacherDashboard() {
    fetch('http://localhost:5000/teacher-stats')
        .then(res => res.json())
        .then(data => {
            // Update stats cards
            updateStatsCards(data);
            // Create charts
            createMoodChart(data.weeklyMoods);
            createCheckinChart(data.dailyCheckins);
            // Setup support popup click handler
            setupSupportPopup(data.studentsNeedSupport);
        })
        .catch(err => console.error('Error loading dashboard:', err));
}

// NEW: Update Stats Cards
function updateStatsCards(data) {
    // Update total students
    document.querySelector('.stat-card:nth-child(1) .stat-number').textContent = data.totalStudents;
    
    // Update most common mood
    document.querySelector('.stat-card:nth-child(2) .stat-number').textContent = data.mostCommonMood;
    
    // Update positive responses percentage
    document.querySelector('.stat-card:nth-child(3) .stat-number').textContent = data.positivePercentage + '%';
    
    // Update students needing support - make it clickable
    const supportCard = document.querySelector('.stat-card:nth-child(4)');
    supportCard.style.cursor = 'pointer';
    supportCard.onclick = () => showSupportPopup(data.studentsNeedSupport);
    supportCard.querySelector('.stat-number').textContent = data.studentsNeedSupport.length;
}

// NEW: Show Support Popup
function showSupportPopup(studentsNeedSupport) {
    const popup = document.getElementById('support-popup');
    const supportList = document.getElementById('support-list');
    
    if (studentsNeedSupport.length === 0) {
        supportList.innerHTML = '<div style="text-align: center; color: #4ade80;">🎉 All students are doing well!</div>';
    } else {
        supportList.innerHTML = studentsNeedSupport.map(student => `
            <div style="background: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #ef4444;">
                <div style="font-weight: 600; color: #ef4444;">${student.name}</div>
                <div style="color: #9ca3af; font-size: 14px;">${student.email}</div>
                <div style="color: #fbbf24; font-size: 14px;">Recent mood: ${student.mood} ${student.moodEmoji}</div>
            </div>
        `).join('');
    }
    
    popup.style.display = 'flex';
}

// NEW: Close Support Popup
function closeSupportPopup() {
    document.getElementById('support-popup').style.display = 'none';
}

// NEW: Setup Support Popup
function setupSupportPopup(studentsNeedSupport) {
    // This function is called from updateStatsCards
}

// NEW: Create Mood Chart
function createMoodChart(weeklyData) {
    const ctx = document.getElementById('moodChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: weeklyData.labels, // ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            datasets: [{
                label: 'Happy 😊',
                data: weeklyData.happy,
                backgroundColor: '#4ade80'
            }, {
                label: 'Neutral 😐',
                data: weeklyData.neutral,
                backgroundColor: '#94a3b8'
            }, {
                label: 'Sad 😢',
                data: weeklyData.sad,
                backgroundColor: '#f87171'
            }, {
                label: 'Anxious 😰',
                data: weeklyData.anxious,
                backgroundColor: '#fbbf24'
            }, {
                label: 'Excited 🤩',
                data: weeklyData.excited,
                backgroundColor: '#a78bfa'
            }, {
                label: 'Tired 😴',
                data: weeklyData.tired,
                backgroundColor: '#64748b'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8' }
                }
            }
        }
    });
}

// NEW: Create Check-in Chart
function createCheckinChart(dailyData) {
    const ctx = document.getElementById('checkinChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dailyData.dates, // ['Dec 15', 'Dec 16', 'Dec 17', ...]
            datasets: [{
                label: 'Daily Check-ins',
                data: dailyData.counts, // [25, 32, 28, 35, ...]
                borderColor: '#4ade80',
                backgroundColor: 'rgba(74, 222, 128, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8' }
                }
            }
        }
    });
}

// Load past chat history from backend
function toggleHistory() {
    const container = document.getElementById('chat-history-container');
    container.classList.toggle('hidden');
}

// Load history into separate container
function loadChatHistory() {
    fetch(`http://localhost:5000/history?email=${currentUser.email}`)
        .then(res => res.json())
        .then(data => {
            const container = document.getElementById('chat-history-list');
            container.innerHTML = '';

            if (data.history && Array.isArray(data.history)) {
                data.history.forEach(entry => {
                    const msgBlock = document.createElement('div');
                    msgBlock.innerHTML = `
                        <div style="color: #90cdf4;"><strong>You:</strong> ${entry.user}</div>
                        <div style="color: #fbcfe8;"><strong>AI:</strong> ${entry.bot}</div>
                        <hr style="border: 0.5px solid rgba(255,255,255,0.1); margin: 8px 0;">
                    `;
                    container.appendChild(msgBlock);
                });
            }
        })
        .catch(err => console.error("Error loading history:", err));
}

// Mood selection - UPDATED TO SEND MOOD TO BACKEND
function selectMood(mood) {
    selectedMood = mood;
    document.querySelectorAll('.mood-btn').forEach(btn => btn.classList.remove('selected'));
    event.target.classList.add('selected');
    document.getElementById('chat-section').classList.remove('hidden');
    
    // Send mood to backend
    fetch('http://localhost:5000/submit-mood', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            email: currentUser.email,
            name: currentUser.username,
            mood: mood,
            timestamp: new Date().toISOString()
        })
    })
    .then(res => res.json())
    .then(data => {
        console.log('Mood submitted successfully:', data);
    })
    .catch(err => console.error('Error submitting mood:', err));
    
    addAIResponse(getMoodResponse(mood));
}

function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (message) {
        addUserMessage(message);
        input.value = '';
        processUserMessage(message);
    }
}

function addUserMessage(message) {
    const chatMessages = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('user-message');
    msgDiv.textContent = message;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addAIResponse(response) {
    const chatMessages = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('ai-response');
    msgDiv.innerHTML = formatAIResponse(response);  
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatAIResponse(text) {
    return DOMPurify.sanitize(marked.parse(text));
}

async function processUserMessage(message) {
    const chatMessages = document.getElementById('chat-messages');
    const loadingDiv = document.createElement('div');
    loadingDiv.textContent = "MindMate is thinking...";
    loadingDiv.style.cssText = 'color: #aaa; font-style: italic; text-align: left; margin-top: 5px;';
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const res = await fetch("http://127.0.0.1:5000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message,
                email: currentUser.email,
                mood: selectedMood
            }),
        });
        const data = await res.json();
        loadingDiv.remove();
        addAIResponse(data.response);
    } catch (err) {
        console.error("Chat API Error:", err);
        loadingDiv.textContent = "❌ Something went wrong. Try again.";
    }
}

// function getMoodResponse(mood) {
//     const responses = {
//         happy: "I'm so glad you're feeling happy today! 😊",
//         sad: "I'm here for you. It's okay to feel sad sometimes. 💙",
//         anxious: "You're feeling anxious. Let's take it one step at a time. 🌸",
//         neutral: "Thanks for checking in. Every feeling is valid. 🌿",
//         excited: "Your excitement is wonderful! ✨",
//         tired: "Rest is important. Be gentle with yourself. 🌙"
//     };
//     return responses[mood] || "Thank you for sharing how you're feeling.";
// }

function clearMessages() {
    document.querySelectorAll('.success-message, .error-message').forEach(m => m.remove());
}

function showMessage(message, type) {
    const messageClass = type === 'success' ? 'success-message' : 'error-message';
    const html = `<div class="${messageClass}">${message}</div>`;
    const activeForm = document.querySelector('.page.active form');
    if (activeForm) {
        activeForm.querySelectorAll('.success-message, .error-message').forEach(m => m.remove());
        activeForm.insertAdjacentHTML('afterbegin', html);
    }
}

function toggleVoiceMode() {
    voiceMode = !voiceMode;
    const btn = document.getElementById('voice-mode-btn');
    const chatInput = document.getElementById('chat-input');

    if (voiceMode) {
        btn.innerHTML = '🛑 Stop Voice Mode';
        btn.classList.add('listening'); // animate
        chatInput.disabled = true; // disable typing
        startVoiceInput();
    } else {
        btn.innerHTML = '🎙 Voice Mode';
        btn.classList.remove('listening');
        chatInput.disabled = false;
        stopVoiceInput();
        stopVoice();  
    }
}


function startVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Speech recognition is not supported in this browser.");
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.continuous = false;

    recognition.start();

    recognition.onstart = () => {
        console.log("🎙 Listening...");
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        processVoiceMessage(transcript);
    };

    recognition.onend = () => {
        if (voiceMode) {
            // Restart automatically if still in voice mode
            recognition.start();
        }
    };

    recognition.onerror = (event) => {
        console.error("Voice input error:", event.error);
    };
}

function stopVoiceInput() {
    if (recognition) {
        recognition.stop();
        recognition = null;
        console.log("🛑 Voice input stopped.");
    }
}


// Process Voice Message (Same as typing)
async function processVoiceMessage(message) {
    addUserMessage(message);
    const chatMessages = document.getElementById('chat-messages');
    const loadingDiv = document.createElement('div');
    loadingDiv.textContent = "MindMate is thinking...";
    loadingDiv.style.cssText = 'color: #aaa; font-style: italic;';
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const res = await fetch("http://127.0.0.1:5000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message,
                email: currentUser.email
            }),
        });
        const data = await res.json();
        loadingDiv.remove();
        addAIResponse(data.response);
        speakText(data.response);
    } catch (err) {
        console.error("Chat API Error:", err);
        loadingDiv.textContent = "❌ Something went wrong. Try again.";
    }
}

// Text-to-Speech
function speakText(text) {
    if (!window.speechSynthesis) return;
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;

    speechSynthesis.speak(utterance);
}
function stopVoice() {
    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        console.log("🔇 Voice stopped.");
    }
}



function logout() {
    currentUser = null;
    selectedMood = null;
    document.getElementById('student-username').value = '';
    document.getElementById('student-password').value = '';
    document.getElementById('teacher-id').value = '';
    document.getElementById('teacher-password').value = '';
    document.getElementById('chat-section').classList.add('hidden');
    document.getElementById('chat-messages').innerHTML = '<div style="color: rgba(255,255,255,0.8); text-align: center; margin-top: 2rem;">Hi! I\'m here to listen and support you. How can I help today?</div>';
    document.querySelectorAll('.mood-btn').forEach(btn => btn.classList.remove('selected'));
    showPage('landing');
}

document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keypress', e => {
            if (e.key === 'Enter') sendMessage();
        });
    }
});