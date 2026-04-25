const form = document.getElementById("detectForm");
const resultCard = document.getElementById("resultCard");
const errorCard = document.getElementById("errorCard");

const aiScore = document.getElementById("aiScore");
const label = document.getElementById("label");
const meta = document.getElementById("meta");
const meterFill = document.getElementById("meterFill");
const classifierProb = document.getElementById("classifierProb");
const heuristicScore = document.getElementById("heuristicScore");
const confidence = document.getElementById("confidence");
const textLength = document.getElementById("textLength");
const detailsJson = document.getElementById("detailsJson");

function showError(message) {
  errorCard.textContent = message;
  errorCard.classList.remove("hidden");
  resultCard.classList.add("hidden");
}

function showResult(data) {
  errorCard.classList.add("hidden");
  resultCard.classList.remove("hidden");

  aiScore.textContent = `${data.ai_score.toFixed(2)}%`;
  label.textContent = data.label;
  meta.textContent = `Source: ${data.extracted_from} • Score range: 0–100`;
  meterFill.style.width = `${Math.max(0, Math.min(100, data.ai_score))}%`;

  classifierProb.textContent = `${data.classifier_prob_ai.toFixed(2)}%`;
  heuristicScore.textContent = `${data.heuristic_score.toFixed(2)}%`;
  confidence.textContent = `${data.confidence.toFixed(2)}%`;
  textLength.textContent = String(data.text_length);
  detailsJson.textContent = JSON.stringify(data.details, null, 2);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(form);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Something went wrong.");
      return;
    }

    showResult(data);
  } catch (err) {
    showError("Failed to contact the server.");
  }
});
