const { text } = require("express");
const languagetool = require("languagetool-api");
const cache = require('../cache/local');

// java -cp languagetool-server.jar org.languagetool.server.HTTPServer --port 8081 --allow-origin

const mod = {

  correctFirstMistake: async (text) => {
    const promise = new Promise((resolve, reject) => {
      languagetool.check({
        language: "de-DE",
        text: text
      }, function (err, res) {
        if (err) {
          return reject(err);
        }
        if (res.matches.length > 0) {
          const match = res.matches[0];
          const start = match.context.offset;
          const end = start + match.context.length;
          const replacement = match.replacements[0]?.value;
          let clone = JSON.parse(JSON.stringify(text));
          if (match.rule.id == "UPPERCASE_SENTENCE_START") { // Deactivated rule
            clone = clone.substring(0, start) + replacement + clone.substring(end);
          } else if (match.rule.id == "KOMMA_ZWISCHEN_HAUPT_UND_NEBENSATZ") {
            if (replacement != undefined && replacement != null) {
              clone = clone.substring(0, start) + replacement + clone.substring(end);
            } else {
              clone = clone.substring(0, match.context.length - match.context.offset) + ", " + clone.substring(match.context.length - match.context.offset + 1);
            }
          } else if (match.rule.id == "GERMAN_SPELLER_RULE") {
            clone = clone.substring(0, start + 1) + replacement + clone.substring(end + 1);
          }
          resolve({ change: true, text: clone, matchSize: match.length });
        } else {
          resolve({ change: false, text: text, matchSize: res.matches.length });
        }
      });
    });
    return promise;
  },

  combineIntoChunks: (text, chunkSize) => {
    let counter = 0;
    let lastText = "";
    const result = [];
    for (let i = 0; i < text.length; i++) {
      if (counter == chunkSize) {
        counter = 0;
        result.push(lastText.slice(1));
        lastText = "";
      }
      lastText += " " + text[i].text;
      counter++;
    }
    if (lastText != "") {
      result.push(lastText.slice(1));
    }
    return result;
  },


  correctText: async (textArr, cb) => {
    const promise = new Promise(async (resolve, reject) => {

      const nextText = mod.combineIntoChunks(textArr, 20);
      const results = [];

      for (let i = 0; i < 5; i++) {
        const text = nextText[i];
        let result = await mod.correctFirstMistake(text);

        for (let j = 0; j < result.matchSize + 5; j++) {
          result = await mod.correctFirstMistake(result.text);
        }
        console.log(i + "/" + nextText.length);
        results.push(result.text);
      }
      resolve(results);
    });

    return promise;
  }

}

module.exports = mod;