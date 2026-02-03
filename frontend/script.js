// DOM Elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const letterEl = document.getElementById("letter");
const confidenceEl = document.getElementById("confidence");
const confidenceBadge = document.getElementById("confidenceBadge");
const wordEl = document.getElementById("word");
const topPredictionsEl = document.getElementById("topPredictions");
const cameraStatus = document.getElementById("cameraStatus");
const videoOverlay = document.getElementById("videoOverlay");
const loader = document.getElementById("loader");
const mainApp = document.getElementById("mainApp");

// Buttons
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const backspaceBtn = document.getElementById("backspaceBtn");
const clearBtn = document.getElementById("clearBtn");
const showSignsBtn = document.getElementById("showSignsBtn");
const wordInput = document.getElementById("wordInput");

// Tab elements
const tabBtns = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

// State
let stream = null;
let intervalId = null;
let currentWord = "";
let latestTopPredictions = [];

// Loader Animation
window.addEventListener("load", () => {
    setTimeout(() => {
        loader.classList.add("hidden");
        setTimeout(() => {
            loader.style.display = "none";
        }, 500);
    }, 1500);
});

// Tab Switching
tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        const targetTab = btn.dataset.tab;
        
        // Update active states
        tabBtns.forEach(b => b.classList.remove("active"));
        tabContents.forEach(c => c.classList.remove("active"));
        
        btn.classList.add("active");
        document.getElementById(targetTab).classList.add("active");
        
        // Stop camera when switching to text-to-sign tab
        if (targetTab === "text-to-sign") {
            stopCamera();
        }
    });
});

// Camera Functions
async function startCamera() {
    if (stream) return;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            videoOverlay.classList.add("hidden");
            cameraStatus.classList.add("active");
            cameraStatus.querySelector("span").textContent = "Active";
            startBtn.querySelector("span").textContent = "⏸";
            startBtn.querySelector("span").nextSibling.textContent = " Pause";
        };
        startSending();
    } catch (err) {
        alert("Could not access camera: " + err.message);
        videoOverlay.classList.remove("hidden");
    }
}

function stopCamera() {
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
    if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
    }
    letterEl.textContent = "—";
    confidenceEl.textContent = "";
    confidenceBadge.style.display = "none";
    videoOverlay.classList.remove("hidden");
    cameraStatus.classList.remove("active");
    cameraStatus.querySelector("span").textContent = "Ready";
    startBtn.querySelector("span").textContent = "▶";
    startBtn.querySelector("span").nextSibling.textContent = " Start Camera";
}

function startSending() {
    // Send a frame every 400ms
    intervalId = setInterval(async () => {
        if (!stream) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(sendFrame, "image/jpeg", 0.85);
    }, 400);
}

async function sendFrame(blob) {
    if (!blob) return;
    const formData = new FormData();
    formData.append("frame", blob, "frame.jpg");

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        handlePrediction(data);
    } catch (err) {
        console.error("Prediction error:", err);
        letterEl.textContent = "—";
        confidenceEl.textContent = "";
        confidenceBadge.style.display = "none";
    }
}

function renderTopPredictions() {
    if (!latestTopPredictions.length) {
        topPredictionsEl.innerHTML = "";
        return;
    }

    topPredictionsEl.innerHTML =
        "<h4>Top Predictions (click to add):</h4><ul>" +
        latestTopPredictions
            .map(
                (p) =>
                    `<li><button type="button" class="pred-btn" data-label="${p.label}">${p.label.toUpperCase()} — ${(p.confidence * 100).toFixed(1)}%</button></li>`
            )
            .join("") +
        "</ul>";
}

function handlePrediction(data) {
    if (!data) return;

    // Only update stored predictions when we have a valid prediction
    if (data.top_predictions && data.prediction) {
        latestTopPredictions = data.top_predictions;
        renderTopPredictions();
    }

    if (!data.prediction) {
        // Keep last predictions visible; just update status text
        letterEl.textContent = "—";
        confidenceEl.textContent = data.message || "No hand detected";
        confidenceBadge.style.display = "none";
        return;
    }

    const { prediction, confidence } = data;
    letterEl.textContent = prediction.toUpperCase();
    confidenceEl.textContent = confidence
        ? `${(confidence * 100).toFixed(1)}%`
        : "";
    confidenceBadge.style.display = "flex";
    
    // Add animation to letter
    letterEl.style.animation = "none";
    setTimeout(() => {
        letterEl.style.animation = "letterPop 0.3s ease-out";
    }, 10);
}

// Event Listeners
startBtn.addEventListener("click", () => {
    if (stream) {
        stopCamera();
    } else {
        startCamera();
    }
});

stopBtn.addEventListener("click", stopCamera);

backspaceBtn.addEventListener("click", () => {
    currentWord = currentWord.slice(0, -1);
    updateWordDisplay();
});

clearBtn.addEventListener("click", () => {
    currentWord = "";
    updateWordDisplay();
});

function updateWordDisplay() {
    if (currentWord) {
        wordEl.innerHTML = currentWord;
        wordEl.querySelector(".placeholder")?.remove();
    } else {
        wordEl.innerHTML = '<span class="placeholder">Your word will appear here...</span>';
    }
}

// Click to choose from top predictions
topPredictionsEl.addEventListener("click", (e) => {
    const btn = e.target.closest(".pred-btn");
    if (!btn) return;

    const label = btn.dataset.label;
    if (!label) return;

    if (label === "space") {
        currentWord += " ";
    } else if (label === "del") {
        currentWord = currentWord.slice(0, -1);
    } else {
        currentWord += label.toUpperCase();
    }

    updateWordDisplay();
});

// Try next image format if current one fails
window.tryNextImage = function(img) {
    const sources = JSON.parse(img.dataset.sources || '[]');
    let currentIndex = parseInt(img.dataset.currentIndex || '0');
    
    if (currentIndex < sources.length - 1) {
        // Try next format
        currentIndex++;
        img.dataset.currentIndex = currentIndex.toString();
        img.src = sources[currentIndex];
    } else {
        // All formats failed, show error
        handleImageError(img, img.dataset.letter || '?');
    }
};

// Image error handler (must be global for inline onerror)
window.handleImageError = function(img, letter) {
    img.style.display = "none";
    const container = img.parentElement;
    container.classList.add("missing");
    container.innerHTML = `
        <div style="padding: 20px; text-align: center; color: var(--danger);">
            <div style="font-size: 24px; margin-bottom: 10px;">⚠️</div>
            <div style="font-weight: 600;">Image not found</div>
            <div style="font-size: 12px; margin-top: 5px; opacity: 0.8;">Add ${letter}.jpg, ${letter}.png, or ${letter}.webp to assets/asl_gifs/</div>
        </div>
    `;
};

// Word to Signs Functionality
function showSignsForWord(word) {
    const signsDisplay = document.getElementById("signsDisplay");
    
    if (!word || word.trim() === "") {
        signsDisplay.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">👐</div>
                <p>Enter a word above to see the sign language sequence</p>
            </div>
        `;
        return;
    }

    // Clean the word - remove spaces and convert to uppercase
    const cleanWord = word.toUpperCase().replace(/\s+/g, "");
    const letters = cleanWord.split("");
    
    if (letters.length === 0) {
        signsDisplay.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">⚠️</div>
                <p>Please enter a valid word</p>
            </div>
        `;
        return;
    }

    // Generate sign items for each letter
    // Try multiple image formats: jpg, png, webp, jpeg
    const imageExtensions = ['jpg', 'png', 'webp', 'jpeg'];
    
    signsDisplay.innerHTML = letters.map((letter, index) => {
        // Handle special cases
        let displayLetter = letter;
        let baseName = "";
        
        if (letter === " ") {
            displayLetter = "SPACE";
            baseName = "space";
        } else if (/[A-Z]/.test(letter)) {
            baseName = letter;
        } else {
            // For numbers or special characters, skip or handle differently
            return null;
        }

        // Create image element that tries multiple formats
        const imageSources = imageExtensions.map(ext => `assets/asl_gifs/${baseName}.${ext}`);
        
        return `
            <div class="sign-item" style="animation-delay: ${index * 0.1}s">
                <div class="sign-letter">${displayLetter}</div>
                <div class="sign-image-container">
                    <img 
                        src="${imageSources[0]}" 
                        alt="ASL sign for ${displayLetter}"
                        class="sign-image"
                        data-letter="${displayLetter}"
                        data-sources='${JSON.stringify(imageSources)}'
                        data-current-index="0"
                        onerror="tryNextImage(this)"
                    />
                </div>
            </div>
        `;
    }).filter(item => item !== null).join("");

    // Add word separator if multiple letters
    if (letters.length > 1) {
        const wordSeparator = document.createElement("div");
        wordSeparator.style.cssText = "width: 100%; text-align: center; padding: 20px; font-size: 24px; font-weight: 700; color: var(--primary);";
        wordSeparator.textContent = `Word: ${cleanWord}`;
        signsDisplay.insertBefore(wordSeparator, signsDisplay.firstChild);
    }
}

showSignsBtn.addEventListener("click", () => {
    const word = wordInput.value.trim();
    showSignsForWord(word);
});

// Allow Enter key to trigger show signs
wordInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        showSignsBtn.click();
    }
});

// Auto-start camera if permissions already granted
navigator.permissions
    .query({ name: "camera" })
    .then((status) => {
        if (status.state === "granted" && document.getElementById("sign-to-text").classList.contains("active")) {
            startCamera();
        }
    })
    .catch(() => {});

// Initialize
updateWordDisplay();
