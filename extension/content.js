// content.js
console.log('Sangeet Premium content script loaded');

// Load Material Icons
const link = document.createElement('link');
link.href = 'https://fonts.googleapis.com/icon?family=Material+Icons';
link.rel = 'stylesheet';
document.head.appendChild(link);

// Function to extract video ID from YouTube URL
function getYouTubeVideoId(url) {
  try {
    const urlObj = new URL(url);
    if (urlObj.hostname === 'www.youtube.com' || urlObj.hostname === 'youtube.com') {
      if (urlObj.pathname === '/watch') {
        return urlObj.searchParams.get('v');
      } else if (urlObj.pathname.startsWith('/embed/')) {
        return urlObj.pathname.split('/')[2];
      }
    } else if (urlObj.hostname === 'youtu.be') {
      return urlObj.pathname.substring(1);
    } else if (urlObj.hostname === 'music.youtube.com') {
      if (urlObj.pathname === '/watch') {
        return urlObj.searchParams.get('v');
      }
    }
    return null;
  } catch (e) {
    console.error('Error getting video ID:', e);
    return null;
  }
}

// Store floating button state in local storage
function saveButtonPosition(left, top) {
  localStorage.setItem('sangeetButtonPosition', JSON.stringify({ left, top }));
}

function getButtonPosition() {
  const position = localStorage.getItem('sangeetButtonPosition');
  return position ? JSON.parse(position) : { left: '20px', top: '20px' };
}

let floatingButton = null;

function createFloatingButton() {
  if (floatingButton) return;

  floatingButton = document.createElement('div');
  floatingButton.id = 'sangeet-floating-button';
  floatingButton.classList.add('sangeet-floating-button');

  const savedPosition = getButtonPosition();
  floatingButton.style.left = savedPosition.left;
  floatingButton.style.top = savedPosition.top;

  floatingButton.innerHTML = `
    <div class="sangeet-button-content">
      <div class="sangeet-equalizer">
        <span></span>
        <span></span>
        <span></span>
      </div>
      <span class="sangeet-button-text">Play in Sangeet</span>
    </div>
  `;

  let isDragging = false;
  let dragOffset = { x: 0, y: 0 };

  // Handle click to play
  floatingButton.addEventListener('click', (e) => {
    if (!isDragging) {
      const videoId = getYouTubeVideoId(window.location.href);
      if (videoId) {
        chrome.storage.local.get(['sangeetUrl'], function(result) {
          const baseUrl = result.sangeetUrl || 'http://127.0.0.1:7800';
          const playUrl = `${baseUrl}/?song=${encodeURIComponent(videoId)}`;
          window.open(playUrl, '_blank');
        });
      }
    }
  });

  // Handle dragging
  floatingButton.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    isDragging = false;
    dragOffset = {
      x: e.clientX - floatingButton.offsetLeft,
      y: e.clientY - floatingButton.offsetTop
    };
    
    const moveHandler = (e) => {
      isDragging = true;
      const left = Math.max(0, Math.min(e.clientX - dragOffset.x, window.innerWidth - floatingButton.offsetWidth));
      const top = Math.max(0, Math.min(e.clientY - dragOffset.y, window.innerHeight - floatingButton.offsetHeight));
      
      floatingButton.style.left = `${left}px`;
      floatingButton.style.top = `${top}px`;
      saveButtonPosition(`${left}px`, `${top}px`);
    };

    const upHandler = () => {
      document.removeEventListener('mousemove', moveHandler);
      document.removeEventListener('mouseup', upHandler);
      setTimeout(() => {
        isDragging = false;
      }, 100);
    };

    document.addEventListener('mousemove', moveHandler);
    document.addEventListener('mouseup', upHandler);
  });

  document.body.appendChild(floatingButton);
}

// Function to show/hide floating button based on video detection
function toggleFloatingButton() {
  const videoId = getYouTubeVideoId(window.location.href);
  if (videoId) {
    if (!floatingButton) {
      createFloatingButton();
    }
    floatingButton.style.display = 'flex';
  } else if (floatingButton) {
    floatingButton.style.display = 'none';
  }
}

// Function to get video information for popup
function getVideoInfo() {
  try {
    const videoId = getYouTubeVideoId(window.location.href);
    const videoTitle = document.querySelector('h1.ytd-video-primary-info-renderer')?.textContent?.trim() || 
                      document.querySelector('h1.title')?.textContent?.trim();
    const channelName = document.querySelector('ytd-video-owner-renderer #channel-name')?.textContent?.trim() ||
                       document.querySelector('#owner-name a')?.textContent?.trim();

    return {
      videoId,
      title: videoTitle || 'Unknown Title',
      channel: channelName || 'Unknown Channel'
    };
  } catch (e) {
    console.error('Error getting video info:', e);
    return null;
  }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getVideoInfo') {
    try {
      const videoInfo = getVideoInfo();
      sendResponse(videoInfo);
    } catch (e) {
      console.error('Error in content script:', e);
      sendResponse(null);
    }
  }
  return true;
});

// Watch for page navigation
let lastUrl = location.href;
new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    lastUrl = url;
    setTimeout(toggleFloatingButton, 1000);
  }
}).observe(document, { subtree: true, childList: true });

// Initial setup
setTimeout(toggleFloatingButton, 1000);