const transcripter = require('youtube-transcript').default;
const cache = require('../cache/local');

const mod = {

  getTranscript: async (videoId) => {
    try {
      const transcript = cache.get(videoId) || await transcripter.fetchTranscript(videoId);
      cache.set(videoId, transcript);
      return transcript;
    } catch (error) {
      console.log(error);
      return "Transcript not available";
    }

  },

  getOnlyText: async (videoId) => {
    try {
      const transcript = cache.get(videoId) || await transcripter.fetchTranscript(videoId);
      cache.set(videoId, transcript);
      return transcript.map(item => item.text);
    } catch (error) {
      console.log(error);
      return "Transcript not available";
    }
  },

  getAsText: async (videoId) => {
    try {
      const textArray = await mod.getOnlyText(videoId);
      const longText = textArray.join(' ');
      const noExtraData = longText.replace(/\[[a-z A-Z 0-9]*\]/gm, '');
      return noExtraData;


    } catch (error) {
      console.log(error);
      return "Transcript not available";
    }
  }

}

module.exports = mod;