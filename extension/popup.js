// popup.js
document.addEventListener('DOMContentLoaded', function() {
  const elements = {
    settingsToggle: document.getElementById('settingsToggle'),
    settingsPanel: document.getElementById('settingsPanel'),
    hostUrl: document.getElementById('hostUrl'),
    saveButton: document.getElementById('saveSettings'),
    playButton: document.getElementById('playButton'),
    downloadButton: document.getElementById('downloadButton'),
    status: document.getElementById('status'),
    noVideoMessage: document.getElementById('noVideoMessage'),
    videoInfo: document.getElementById('videoInfo'),
    videoTitle: document.getElementById('videoTitle'),
    channelName: document.getElementById('channelName')
  };

  // Show status message
  function showStatus(message) {
    if (!elements.status) return;
    elements.status.textContent = message;
    setTimeout(() => {
      elements.status.textContent = '';
    }, 3000);
  }

  // Check current tab and update UI
  async function checkCurrentTab() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const isYouTubeVideo = tab?.url?.includes('youtube.com/watch?v=');

      if (elements.playButton) {
        elements.playButton.classList.toggle('active', isYouTubeVideo);
      }
      if (elements.downloadButton) {
        elements.downloadButton.classList.toggle('active', isYouTubeVideo);
      }
      if (elements.noVideoMessage) {
        elements.noVideoMessage.classList.toggle('visible', !isYouTubeVideo);
      }
      if (elements.videoInfo) {
        elements.videoInfo.classList.toggle('active', isYouTubeVideo);
      }

      if (isYouTubeVideo) {
        try {
          const response = await new Promise((resolve) => {
            chrome.tabs.sendMessage(tab.id, { action: 'getVideoInfo' }, (response) => {
              if (chrome.runtime.lastError) {
                console.log('Error:', chrome.runtime.lastError);
                resolve(null);
              } else {
                resolve(response);
              }
            });
          });

          if (response) {
            if (elements.videoTitle) {
              elements.videoTitle.textContent = response.title || 'Unknown Title';
            }
            if (elements.channelName) {
              elements.channelName.textContent = response.channel || 'Unknown Channel';
            }
          }
        } catch (error) {
          console.error('Error getting video info:', error);
        }
      }
    } catch (error) {
      console.error('Error checking current tab:', error);
    }
  }

  // Load settings
  function loadSettings() {
    chrome.storage.local.get(['sangeetUrl'], function(result) {
      if (elements.hostUrl) {
        elements.hostUrl.value = result.sangeetUrl || 'http://127.0.0.1:7800';
      }
    });
  }

  // Initialize
  loadSettings();
  checkCurrentTab();

  // Event Listeners
  if (elements.settingsToggle) {
    elements.settingsToggle.addEventListener('click', () => {
      if (elements.settingsPanel) {
        elements.settingsPanel.classList.toggle('visible');
      }
    });
  }

  if (elements.saveButton) {
    elements.saveButton.addEventListener('click', () => {
      if (!elements.hostUrl) return;
      
      const sangeetUrl = elements.hostUrl.value.trim();
      
      if (!sangeetUrl) {
        showStatus('Please enter a valid host URL');
        return;
      }

      try {
        new URL(sangeetUrl);
      } catch {
        showStatus('Please enter a valid URL');
        return;
      }

      chrome.storage.local.set({ sangeetUrl }, function() {
        if (chrome.runtime.lastError) {
          showStatus('Error saving settings: ' + chrome.runtime.lastError.message);
        } else {
          showStatus('Settings saved successfully!');
          setTimeout(() => {
            if (elements.settingsPanel) {
              elements.settingsPanel.classList.remove('visible');
            }
          }, 1500);
        }
      });
    });
  }

  // Play button handler
  if (elements.playButton) {
    elements.playButton.addEventListener('click', async () => {
      try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab?.url?.includes('youtube.com/watch?v=')) {
          const response = await new Promise((resolve) => {
            chrome.tabs.sendMessage(tab.id, { action: 'getVideoInfo' }, (response) => {
              if (chrome.runtime.lastError) {
                resolve(null);
              } else {
                resolve(response);
              }
            });
          });

          if (response?.videoId) {
            const result = await chrome.storage.local.get(['sangeetUrl']);
            const baseUrl = result.sangeetUrl || 'http://127.0.0.1:7800';
            const playUrl = `${baseUrl}/?song=${encodeURIComponent(response.videoId)}`;
            await chrome.tabs.create({ url: playUrl });
          }
        }
      } catch (error) {
        console.error('Error playing video:', error);
        showStatus('Error playing video. Please try again.');
      }
    });
  }

  // Download button handler
  if (elements.downloadButton) {
    elements.downloadButton.addEventListener('click', async () => {
      try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab?.url?.includes('youtube.com/watch?v=')) {
          const response = await new Promise((resolve) => {
            chrome.tabs.sendMessage(tab.id, { action: 'getVideoInfo' }, (response) => {
              if (chrome.runtime.lastError) {
                resolve(null);
              } else {
                resolve(response);
              }
            });
          });

          if (response?.videoId) {
            const result = await chrome.storage.local.get(['sangeetUrl']);
            const baseUrl = result.sangeetUrl || 'http://127.0.0.1:7800';
            const downloadUrl = `${baseUrl}/sangeet-download/${encodeURIComponent(response.videoId)}`;
            await chrome.tabs.create({ url: downloadUrl });
          }
        }
      } catch (error) {
        console.error('Error downloading video:', error);
        showStatus('Error downloading song. Please try again.');
      }
    });
  }
});