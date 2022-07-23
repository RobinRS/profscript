const express = require("express");
const youtube = require("./src/core/youtube");
const cleaner = require("./src/core/cleaner");

const app = express();

app.get("/", (req, res) => {
  res.send("This is home page.");
});

app.get("/transcript/:videoId", async (req, res) => {
  const transcript = await youtube.getTranscript(req.params.videoId);
  const correctedText = await cleaner.correctText(transcript);
  res.send(correctedText);
});

app.get("/trans/:videoId", async (req, res) => {
  const transcript = await youtube.getAsText(req.params.videoId);
  res.send(transcript);
});


// PORT
const PORT = 3000;

app.listen(PORT, () => {
  console.log(`Server is running on PORT: ${PORT}`);
});
