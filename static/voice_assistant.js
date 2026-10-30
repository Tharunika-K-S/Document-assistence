/**
 * GovEase AI — Bilingual Voice Assistant (TTS & STT)
 * Supports English & Tamil with native browser SpeechSynthesis & SpeechRecognition,
 * multi-stream prevention, audio chunking for long text, live transcripts, and fallbacks.
 */

(function () {
    'use strict';

    // Singleton Voice Controller
    class VoiceAssistantController {
        constructor() {
            this.synth = window.speechSynthesis || null;
            this.recognition = null;
            this.voices = [];
            this.currentLanguage = 'en'; // 'en' or 'ta'
            this.contextType = 'document'; // 'document' or 'scheme'

            // Playback State
            this.isPlaying = false;
            this.isPaused = false;
            this.activeUtterance = null;
            this.audioChunks = [];
            this.currentChunkIndex = 0;
            this.currentTitle = '';
            this.activeSourceBtn = null;

            // STT State
            this.isListening = false;
            this.recognizedText = '';

            this.init();
        }

        init() {
            // 1. Detect Context (Document page vs Scheme page)
            if (window.location.pathname.includes('/schemes')) {
                this.contextType = 'scheme';
            } else {
                this.contextType = 'document';
            }

            // 2. Initialize Voices for TTS
            if (this.synth) {
                this.loadVoices();
                if (typeof this.synth.onvoiceschanged !== 'undefined') {
                    this.synth.onvoiceschanged = () => this.loadVoices();
                }
            }

            // 3. Initialize SpeechRecognition for STT
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition || null;
            if (SpeechRec) {
                try {
                    this.recognition = new SpeechRec();
                    this.recognition.continuous = false;
                    this.recognition.interimResults = true;
                    this.recognition.maxAlternatives = 1;
                    this.setupRecognitionEvents();
                } catch (e) {
                    console.warn('[GovEase Voice] SpeechRecognition init failed:', e);
                    this.recognition = null;
                }
            }

            // 4. Build and Mount UI Elements
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => this.mountUI());
            } else {
                this.mountUI();
            }
        }

        loadVoices() {
            if (!this.synth) return;
            this.voices = this.synth.getVoices() || [];
        }

        isTamilText(str) {
            return /[\u0B80-\u0BFF]/.test(str || '');
        }

        getVoiceForLanguage(lang, text) {
            if (!this.voices || this.voices.length === 0) {
                this.loadVoices();
            }

            const needsTamil = (lang === 'ta' || this.isTamilText(text));

            if (needsTamil) {
                // Find Tamil voice
                const taVoice = this.voices.find(v =>
                    v.lang === 'ta-IN' || v.lang === 'ta_IN' || v.lang.startsWith('ta') ||
                    (v.name && v.name.toLowerCase().includes('tamil')) ||
                    (v.name && v.name.toLowerCase().includes('valluvar'))
                );
                if (taVoice) return taVoice;
            }

            // Find English voice (prefer Indian English if available for natural bilingual accent)
            const enInVoice = this.voices.find(v => v.lang === 'en-IN' || v.lang === 'en_IN');
            if (enInVoice && !needsTamil) return enInVoice;

            const enVoice = this.voices.find(v => v.lang.startsWith('en'));
            if (enVoice && !needsTamil) return enVoice;

            return this.voices[0] || null;
        }

        // ========================================================
        // TEXT-TO-SPEECH (TTS) ENGINE
        // ========================================================

        /**
         * Splits long text into natural sentence/comma chunks under 200 chars
         * to avoid browser speech synthesis freezes.
         */
        chunkText(text) {
            if (!text) return [];
            const clean = text.replace(/[\r\n]+/g, ' ').replace(/\s+/g, ' ').trim();
            const rawSentences = clean.match(/[^.!?]+[.!?]+|\S+/g) || [clean];
            const chunks = [];
            let cur = '';

            rawSentences.forEach(s => {
                if ((cur + ' ' + s).length < 180) {
                    cur += (cur ? ' ' : '') + s;
                } else {
                    if (cur) chunks.push(cur.trim());
                    cur = s;
                }
            });
            if (cur) chunks.push(cur.trim());
            return chunks.filter(c => c.length > 0);
        }

        /**
         * Reads the provided text aloud with full Play/Pause/Stop control.
         * Guarantees only one audio stream plays at any time.
         */
        playText(text, lang, title = 'Reading Aloud', sourceBtn = null) {
            if (!this.synth) {
                alert('Your browser does not support Speech Synthesis (Text-to-Speech).');
                return;
            }

            // Prevent multiple audio streams: Cancel any existing utterance
            this.stop();

            if (!text || !text.trim()) return;

            // Detect language
            const actualLang = (lang === 'ta' || this.isTamilText(text)) ? 'ta' : (lang || this.currentLanguage);
            this.audioChunks = this.chunkText(text);
            this.currentChunkIndex = 0;
            this.currentTitle = title;
            this.currentLanguage = actualLang;

            // Highlight source button if provided
            if (this.activeSourceBtn) {
                this.activeSourceBtn.classList.remove('active');
            }
            if (sourceBtn) {
                this.activeSourceBtn = sourceBtn;
                this.activeSourceBtn.classList.add('active');
            }

            this.isPlaying = true;
            this.isPaused = false;
            this.updateAudioBarUI('speaking');
            this.speakNextChunk();
        }

        speakNextChunk() {
            if (!this.isPlaying || this.isPaused) return;

            if (this.currentChunkIndex >= this.audioChunks.length) {
                // Finished reading
                this.onPlaybackEnded();
                return;
            }

            const chunkText = this.audioChunks[this.currentChunkIndex];
            const utterance = new SpeechSynthesisUtterance(chunkText);
            const voice = this.getVoiceForLanguage(this.currentLanguage, chunkText);

            if (voice) {
                utterance.voice = voice;
                utterance.lang = voice.lang;
            } else {
                utterance.lang = (this.currentLanguage === 'ta' || this.isTamilText(chunkText)) ? 'ta-IN' : 'en-US';
            }

            utterance.rate = (utterance.lang.startsWith('ta')) ? 0.95 : 1.0;
            utterance.pitch = 1.0;

            utterance.onend = () => {
                this.currentChunkIndex++;
                this.speakNextChunk();
            };

            utterance.onerror = (e) => {
                if (e.error === 'interrupted' || e.error === 'canceled') return;
                console.warn('[GovEase Voice] TTS chunk error:', e);
                this.currentChunkIndex++;
                this.speakNextChunk();
            };

            this.activeUtterance = utterance;
            this.synth.speak(utterance);
        }

        pause() {
            if (!this.synth || !this.isPlaying || this.isPaused) return;
            this.synth.pause();
            this.isPaused = true;
            this.updateAudioBarUI('paused');
        }

        resume() {
            if (!this.synth || !this.isPlaying || !this.isPaused) return;
            this.synth.resume();
            this.isPaused = false;
            this.updateAudioBarUI('speaking');
        }

        stop() {
            if (this.synth) {
                this.synth.cancel();
            }
            this.isPlaying = false;
            this.isPaused = false;
            this.activeUtterance = null;
            this.audioChunks = [];
            this.currentChunkIndex = 0;

            if (this.activeSourceBtn) {
                this.activeSourceBtn.classList.remove('active');
                this.activeSourceBtn = null;
            }

            this.updateAudioBarUI('stopped');
        }

        onPlaybackEnded() {
            this.stop();
        }

        // ========================================================
        // SPEECH-TO-TEXT (STT) ENGINE
        // ========================================================

        setupRecognitionEvents() {
            if (!this.recognition) return;

            this.recognition.onstart = () => {
                this.isListening = true;
                this.updateMicStatusUI('listening');
            };

            this.recognition.onresult = (event) => {
                let interim = '';
                let final = '';

                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        final += event.results[i][0].transcript;
                    } else {
                        interim += event.results[i][0].transcript;
                    }
                }

                const transcriptBox = document.getElementById('voiceTranscript');
                if (transcriptBox) {
                    transcriptBox.innerHTML = final
                        ? `<strong>${this.escapeHtml(final)}</strong>`
                        : `<span class="interim">${this.escapeHtml(interim || 'Listening...')}</span>`;
                }

                if (final) {
                    this.recognizedText = final.trim();
                    const inputField = document.getElementById('voiceTextInput');
                    if (inputField) {
                        inputField.value = this.recognizedText;
                    }
                }
            };

            this.recognition.onerror = (event) => {
                console.warn('[GovEase Voice] Recognition error:', event.error);
                let msg = 'Voice recognition error. You can type your question below.';
                if (event.error === 'not-allowed' || event.error === 'permission-denied') {
                    msg = 'Microphone permission was denied. Please allow microphone access or type your question below.';
                } else if (event.error === 'no-speech') {
                    msg = 'No speech detected. Please try clicking the microphone again or type below.';
                }

                this.showErrorBanner(msg);
                this.isListening = false;
                this.updateMicStatusUI('error');
            };

            this.recognition.onend = () => {
                this.isListening = false;
                this.updateMicStatusUI('idle');

                // If text was recognized, auto ask after a brief pause
                if (this.recognizedText) {
                    const textToSend = this.recognizedText;
                    this.recognizedText = '';
                    this.sendQuestion(textToSend);
                }
            };
        }

        startListening() {
            if (!this.recognition) {
                this.showErrorBanner('Voice input is not supported in this browser. Please type your question in the text box below.');
                const inputField = document.getElementById('voiceTextInput');
                if (inputField) inputField.focus();
                return;
            }

            // Stop any ongoing speech output so mic does not hear speaker
            this.stop();
            this.hideErrorBanner();

            try {
                this.recognition.lang = (this.currentLanguage === 'ta') ? 'ta-IN' : 'en-IN';
                this.recognizedText = '';
                const transcriptBox = document.getElementById('voiceTranscript');
                if (transcriptBox) {
                    transcriptBox.innerHTML = '<span class="interim">🎙️ Listening... Speak now</span>';
                }
                this.recognition.start();
            } catch (err) {
                console.warn('[GovEase Voice] Could not start recognition:', err);
            }
        }

        stopListening() {
            if (this.recognition && this.isListening) {
                this.recognition.stop();
            }
        }

        toggleListening() {
            if (this.isListening) {
                this.stopListening();
            } else {
                this.startListening();
            }
        }

        // ========================================================
        // ASYNC QUESTION ANSWERING
        // ========================================================

        async sendQuestion(questionText) {
            const q = (questionText || '').trim();
            if (!q) return;

            // Add user message to chat history
            this.appendChatMessage('user', q);
            this.updateMicStatusUI('processing');

            const transcriptBox = document.getElementById('voiceTranscript');
            if (transcriptBox) {
                transcriptBox.innerHTML = '<span class="interim">Thinking / பதிலளிக்கிறது...</span>';
            }

            try {
                const response = await fetch('/api/voice/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: q,
                        context_type: this.contextType,
                        lang: this.currentLanguage
                    })
                });

                const data = await response.json();
                if (data.success && data.answer) {
                    this.appendChatMessage('assistant', data.answer, data.lang || this.currentLanguage);
                    // Automatically read the assistant's answer aloud
                    this.playText(data.answer, data.lang || this.currentLanguage, 'Assistant Answer');
                } else {
                    const fallback = data.error || 'Could not process question. Please try asking in a different way.';
                    this.appendChatMessage('assistant', fallback, 'en');
                }
            } catch (err) {
                console.error('[GovEase Voice] QA request error:', err);
                const errMsg = (this.currentLanguage === 'ta')
                    ? 'மன்னிக்கவும், பதிலைப் பெறுவதில் சிக்கல் ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.'
                    : 'Sorry, there was a temporary network issue. Please try again.';
                this.appendChatMessage('assistant', errMsg, this.currentLanguage);
            } finally {
                this.updateMicStatusUI('idle');
                const transcriptBox = document.getElementById('voiceTranscript');
                if (transcriptBox) {
                    transcriptBox.innerHTML = '<span class="interim">Click microphone to speak or type a question below</span>';
                }
            }
        }

        // ========================================================
        // UI MOUNTING & DOM INJECTION
        // ========================================================

        mountUI() {
            // 1. Inject Floating Audio Bar
            if (!document.getElementById('voiceAudioBar')) {
                const bar = document.createElement('aside');
                bar.id = 'voiceAudioBar';
                bar.className = 'voice-audio-bar';
                bar.setAttribute('aria-label', 'Audio Playback Controls');
                bar.innerHTML = `
                    <div class="voice-bar-info">
                        <span class="voice-bar-title" id="voiceBarTitle">GovEase Audio</span>
                        <span class="voice-bar-status">
                            <span class="voice-status-dot" id="voiceStatusDot"></span>
                            <span id="voiceBarStatusText">Ready</span>
                        </span>
                    </div>
                    <div class="voice-bar-controls">
                        <button class="voice-ctrl-btn primary" id="voicePlayPauseBtn" title="Pause / Resume" onclick="window.GovEaseVoice.togglePlayPause()">⏸</button>
                        <button class="voice-ctrl-btn stop" id="voiceStopBtn" title="Stop Audio" onclick="window.GovEaseVoice.stop()">⏹</button>
                    </div>
                    <button class="voice-bar-close" title="Hide Audio Bar" onclick="window.GovEaseVoice.hideAudioBar()">✕</button>
                `;
                document.body.appendChild(bar);
            }

            // 2. Inject Floating Action Button (FAB)
            if (!document.getElementById('voiceFab')) {
                const fab = document.createElement('button');
                fab.id = 'voiceFab';
                fab.className = 'voice-fab';
                fab.title = 'Open GovEase Voice Assistant (குரல் உதவியாளர்)';
                fab.setAttribute('aria-label', 'Open Voice Assistant');
                fab.innerHTML = `🎙️<span class="voice-fab-badge">AI</span>`;
                fab.onclick = () => this.openModal();
                document.body.appendChild(fab);
            }

            // 3. Inject Voice Assistant Modal / Drawer
            if (!document.getElementById('voiceModalBackdrop')) {
                const modal = document.createElement('div');
                modal.id = 'voiceModalBackdrop';
                modal.className = 'voice-modal-backdrop';
                modal.innerHTML = `
                    <div class="voice-modal" role="dialog" aria-modal="true" aria-labelledby="voiceModalTitle">
                        <div class="voice-modal-header">
                            <div class="voice-modal-title" id="voiceModalTitle">
                                <span>🎙️</span>
                                <div>
                                    <div>GovEase Voice Assistant</div>
                                    <div style="font-size:11px; opacity:0.8; font-weight:400;">குரல் உதவியாளர் (Bilingual QA)</div>
                                </div>
                            </div>
                            <div style="display:flex; align-items:center; gap:10px;">
                                <button class="lang-toggle-btn" id="voiceLangToggleBtn" onclick="window.GovEaseVoice.toggleLanguage()">
                                    Language: <strong id="voiceLangLabel">EN</strong>
                                </button>
                                <button class="voice-modal-close" onclick="window.GovEaseVoice.closeModal()" aria-label="Close">✕</button>
                            </div>
                        </div>

                        <div class="voice-modal-body">
                            <!-- Alert Banner for errors / permissions -->
                            <div class="voice-alert-banner" id="voiceAlertBanner">
                                <span>⚠️</span>
                                <span id="voiceAlertText">Microphone notice</span>
                            </div>

                            <!-- Live Microphone Status -->
                            <div class="voice-mic-status-box" id="voiceMicStatusBox">
                                <button class="voice-mic-circle" id="voiceMicCircle" onclick="window.GovEaseVoice.toggleListening()" aria-label="Microphone">
                                    🎙️
                                </button>
                                <div class="voice-status-text" id="voiceMicStatusText">Click Mic to Speak</div>
                                <div class="voice-status-sub" id="voiceMicStatusSub">Speak your question in English or தமிழ்</div>
                                <div class="voice-wave-bars">
                                    <div class="voice-wave-bar"></div>
                                    <div class="voice-wave-bar"></div>
                                    <div class="voice-wave-bar"></div>
                                    <div class="voice-wave-bar"></div>
                                    <div class="voice-wave-bar"></div>
                                </div>
                            </div>

                            <!-- Real-time Live Recognized Speech Bubble -->
                            <div class="voice-transcript-box" id="voiceTranscript">
                                <span class="interim">Click microphone to speak or type a question below</span>
                            </div>

                            <!-- Contextual Question Suggestion Chips -->
                            <div>
                                <div class="voice-chips-label">Suggested Questions / கேட்கக்கூடியவை:</div>
                                <div class="voice-chips-row" id="voiceChipsRow">
                                    <!-- Populated dynamically based on context -->
                                </div>
                            </div>

                            <!-- Conversation History -->
                            <div class="voice-chat-history" id="voiceChatHistory"></div>
                        </div>

                        <div class="voice-modal-footer">
                            <input type="text" id="voiceTextInput" class="voice-text-input" placeholder="Type a question about document or schemes..." />
                            <button class="voice-send-btn" id="voiceSendBtn" onclick="window.GovEaseVoice.submitTextInput()">Ask</button>
                        </div>
                    </div>
                `;
                document.body.appendChild(modal);

                // Handle outside click to close
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) this.closeModal();
                });

                // Handle Enter key on input
                const textInput = document.getElementById('voiceTextInput');
                if (textInput) {
                    textInput.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter') {
                            this.submitTextInput();
                        }
                    });
                }
            }

            this.updateSuggestionChips();
            this.syncPageLanguage();
        }

        // ========================================================
        // UI HELPERS & STATE MANAGEMENT
        // ========================================================

        openModal() {
            const backdrop = document.getElementById('voiceModalBackdrop');
            if (backdrop) {
                backdrop.classList.add('open');
            }
        }

        closeModal() {
            this.stopListening();
            const backdrop = document.getElementById('voiceModalBackdrop');
            if (backdrop) {
                backdrop.classList.remove('open');
            }
        }

        toggleLanguage() {
            this.currentLanguage = (this.currentLanguage === 'en') ? 'ta' : 'en';
            const label = document.getElementById('voiceLangLabel');
            if (label) {
                label.innerText = (this.currentLanguage === 'en') ? 'EN' : 'தமிழ்';
            }
            this.updateSuggestionChips();
        }

        setLanguage(lang) {
            this.currentLanguage = (lang === 'ta') ? 'ta' : 'en';
            const label = document.getElementById('voiceLangLabel');
            if (label) {
                label.innerText = (this.currentLanguage === 'en') ? 'EN' : 'தமிழ்';
            }
            this.updateSuggestionChips();
        }

        syncPageLanguage() {
            // Check if page already has active language toggle button
            const activeLangBtn = document.querySelector('.lang-btn.active');
            if (activeLangBtn) {
                const isTa = activeLangBtn.textContent.includes('தமிழ்');
                this.setLanguage(isTa ? 'ta' : 'en');
            }
        }

        updateSuggestionChips() {
            const chipsRow = document.getElementById('voiceChipsRow');
            if (!chipsRow) return;

            const isTa = (this.currentLanguage === 'ta');
            let suggestions = [];

            if (this.contextType === 'scheme') {
                suggestions = isTa ? [
                    'விவசாய திட்டங்கள் என்ன?',
                    'மகளிர் உதவித்தொகை உள்ளதா?',
                    'மாணவர் கல்வி உதவித்தொகை',
                    'முதியோர் ஓய்வூதிய திட்டம்'
                ] : [
                    'Schemes for farmers',
                    'Scholarships for students',
                    'Schemes for women',
                    'Senior citizen pensions'
                ];
            } else {
                suggestions = isTa ? [
                    'இந்த ஆவணத்தின் சுருக்கம் என்ன?',
                    'அடுத்த கட்ட நடவடிக்கை என்ன?',
                    'ஆவணத்தில் உள்ள தேதி என்ன?',
                    'மனுதாரர் பெயர் யார்?'
                ] : [
                    'What is this document about?',
                    'What are my next action steps?',
                    'When was this document issued?',
                    'Who is the applicant or petitioner?'
                ];
            }

            chipsRow.innerHTML = suggestions.map(q => `
                <button class="voice-chip" onclick="window.GovEaseVoice.askSuggested('${this.escapeHtml(q)}')">${this.escapeHtml(q)}</button>
            `).join('');
        }

        askSuggested(q) {
            const inputField = document.getElementById('voiceTextInput');
            if (inputField) inputField.value = q;
            this.sendQuestion(q);
        }

        submitTextInput() {
            const inputField = document.getElementById('voiceTextInput');
            if (!inputField) return;
            const text = inputField.value.trim();
            if (text) {
                inputField.value = '';
                this.sendQuestion(text);
            }
        }

        appendChatMessage(role, text, lang) {
            const chatHistory = document.getElementById('voiceChatHistory');
            if (!chatHistory) return;

            const msgDiv = document.createElement('div');
            msgDiv.className = `voice-msg ${role}`;

            if (role === 'assistant') {
                msgDiv.innerHTML = `
                    <div>${this.escapeHtml(text)}</div>
                    <div class="voice-msg-actions">
                        <button class="voice-read-btn" style="padding:2px 8px; font-size:11px;" onclick="window.GovEaseVoice.playText('${this.escapeHtml(text).replace(/'/g, "\\'")}', '${lang || this.currentLanguage}', 'Assistant Answer', this)">
                            🔊 Listen
                        </button>
                    </div>
                `;
            } else {
                msgDiv.innerText = text;
            }

            chatHistory.appendChild(msgDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }

        updateMicStatusUI(status) {
            const circle = document.getElementById('voiceMicCircle');
            const statusText = document.getElementById('voiceMicStatusText');
            const statusBox = document.getElementById('voiceMicStatusBox');
            const fab = document.getElementById('voiceFab');

            if (status === 'listening') {
                if (circle) circle.classList.add('listening');
                if (statusBox) statusBox.classList.add('listening');
                if (fab) fab.classList.add('listening');
                if (statusText) statusText.innerText = (this.currentLanguage === 'ta') ? 'கேட்கிறது... பேசுங்கள்' : 'Listening... Speak now';
            } else if (status === 'processing') {
                if (circle) circle.classList.remove('listening');
                if (statusBox) statusBox.classList.remove('listening');
                if (fab) fab.classList.remove('listening');
                if (statusText) statusText.innerText = (this.currentLanguage === 'ta') ? 'செயலாக்குகிறது...' : 'Processing your question...';
            } else {
                if (circle) circle.classList.remove('listening');
                if (statusBox) statusBox.classList.remove('listening');
                if (fab) fab.classList.remove('listening');
                if (statusText) statusText.innerText = (this.currentLanguage === 'ta') ? 'பேச மைக்-ஐ அழுத்தவும்' : 'Click Mic to Speak';
            }
        }

        updateAudioBarUI(state) {
            const bar = document.getElementById('voiceAudioBar');
            const titleEl = document.getElementById('voiceBarTitle');
            const statusEl = document.getElementById('voiceBarStatusText');
            const dotEl = document.getElementById('voiceStatusDot');
            const ppBtn = document.getElementById('voicePlayPauseBtn');

            if (!bar) return;

            if (state === 'speaking') {
                bar.classList.add('visible');
                if (titleEl) titleEl.innerText = this.currentTitle || 'Reading Aloud';
                if (statusEl) statusEl.innerText = 'Speaking';
                if (dotEl) { dotEl.className = 'voice-status-dot speaking'; }
                if (ppBtn) { ppBtn.innerText = '⏸'; ppBtn.title = 'Pause'; }
            } else if (state === 'paused') {
                bar.classList.add('visible');
                if (statusEl) statusEl.innerText = 'Paused';
                if (dotEl) { dotEl.className = 'voice-status-dot paused'; }
                if (ppBtn) { ppBtn.innerText = '▶'; ppBtn.title = 'Resume'; }
            } else {
                // Stopped
                if (dotEl) { dotEl.className = 'voice-status-dot'; }
                if (statusEl) statusEl.innerText = 'Stopped';
                if (ppBtn) { ppBtn.innerText = '▶'; }
                setTimeout(() => {
                    if (!this.isPlaying) bar.classList.remove('visible');
                }, 1800);
            }
        }

        togglePlayPause() {
            if (!this.isPlaying) return;
            if (this.isPaused) {
                this.resume();
            } else {
                this.pause();
            }
        }

        hideAudioBar() {
            this.stop();
            const bar = document.getElementById('voiceAudioBar');
            if (bar) bar.classList.remove('visible');
        }

        showErrorBanner(msg) {
            const banner = document.getElementById('voiceAlertBanner');
            const text = document.getElementById('voiceAlertText');
            if (banner && text) {
                text.innerText = msg;
                banner.classList.add('visible');
            }
        }

        hideErrorBanner() {
            const banner = document.getElementById('voiceAlertBanner');
            if (banner) banner.classList.remove('visible');
        }

        escapeHtml(str) {
            if (!str) return '';
            return str
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        // ========================================================
        // DOM READ ALOUD HELPERS (DOCUMENT & SCHEME PAGES)
        // ========================================================

        /**
         * Reads content of an element specified by CSS selector or direct DOM element.
         */
        readElement(selectorOrEl, title = 'Document Content', sourceBtn = null) {
            let el = (typeof selectorOrEl === 'string') ? document.querySelector(selectorOrEl) : selectorOrEl;
            if (!el) return;

            // Extract visible text, stripping buttons and icons
            const clone = el.cloneNode(true);
            const buttons = clone.querySelectorAll('button, script, style, .voice-read-btn');
            buttons.forEach(b => b.remove());

            const text = clone.innerText || clone.textContent || '';
            const isTa = this.isTamilText(text);
            this.playText(text, isTa ? 'ta' : 'en', title, sourceBtn);
        }

        /**
         * Reads a specific scheme card from window.currentResults on schemes.html
         */
        readSchemeCard(index, sourceBtn = null) {
            if (!window.currentResults || !window.currentResults[index]) return;
            const s = window.currentResults[index];
            const name = s.name || 'Scheme';
            const state = s.state ? `State: ${s.state}` : 'Central All India initiative';
            const desc = s.description || '';
            const match = s.match_score ? `${s.match_score} percent match.` : '';
            const reasons = (s.reasons && s.reasons.length) ? `Why it matched: ${s.reasons.join('. ')}` : '';

            const speechText = `${name}. ${state}. ${match} ${desc}. ${reasons}`;
            this.playText(speechText, this.isTamilText(speechText) ? 'ta' : 'en', `Scheme: ${name}`, sourceBtn);
        }
    }

    // Initialize Global Singleton
    window.GovEaseVoice = new VoiceAssistantController();

})();
