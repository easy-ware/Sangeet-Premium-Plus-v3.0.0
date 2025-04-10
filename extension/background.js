// background.js
chrome.runtime.onInstalled.addListener(() => {
  // Set default settings
  const defaultSettings = {
    sangeetUrl: 'http://localhost:7800'
  };

  chrome.storage.local.get(['sangeetUrl'], function(result) {
    if (!result.sangeetUrl) {
      chrome.storage.local.set(defaultSettings);
    }
  });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'openSangeet') {
    const videoId = request.videoId;
    if (!videoId) {
      console.error('No video ID provided');
      return;
    }

    chrome.storage.local.get(['sangeetUrl'], function(result) {
      const baseUrl = result.sangeetUrl || 'http://localhost:7800';
      const redirectUrl = `${baseUrl}/?song=${encodeURIComponent(videoId)}`;
      chrome.tabs.create({ url: redirectUrl });
    });
  }
});