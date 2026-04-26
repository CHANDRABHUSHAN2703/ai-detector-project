const form = document.getElementById("detectForm");
const loading = document.getElementById("loading");
const resultBox = document.getElementById("resultBox");
const errorBox = document.getElementById("errorBox");

const overallLabel = document.getElementById("overallLabel");
const overallScore = document.getElementById("overallScore");
const statsBox = document.getElementById("stats");
const signalsBox = document.getElementById("signals");
const sentenceResultsBox = document.getElementById("sentenceResults");
const themeToggle = document.getElementById("themeToggle");

function setTheme(theme) {
  document.body.classList.toggle("dark", theme === "dark");
  themeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
  localStorage.setItem("theme", theme);
}

function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved === "dark" || saved === "light") {
    setTheme(saved);
  } else {
    setTheme(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }
}

themeToggle.addEventListener("click", () => {
  const nextTheme = document.body.classList.contains("dark") ? "light" : "dark";
  setTheme(nextTheme);
});

initTheme();

function showError(message) {
  errorBox.textContent = message || "Something went wrong.";
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function scoreToPercent(score) {
  return Math.round(score * 100);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  resultBox.classList.add("hidden");
  loading.classList.remove("hidden");

  const formData = new FormData(form);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!data.success) {
      showError(data.error || "Something went wrong");
      return;
    }

    const result = data.result;

    overallLabel.textContent = result.label;
    overallScore.textContent = `${scoreToPercent(result.overall_score)}% AI Score`;

    overallLabel.className = "badge " + (
      result.label === "Likely AI" ? "ai" :
      result.label === "Mixed" ? "mixed" :
      "human"
    );

    const stats = result.stats || {};
    statsBox.innerHTML = `
      <p><strong>Words:</strong> ${stats.word_count ?? 0}</p>
      <p><strong>Sentences:</strong> ${stats.sentence_count ?? 0}</p>
      <p><strong>Avg Sentence Length:</strong> ${stats.avg_sentence_length ?? 0}</p>
      <p><strong>Lexical Diversity:</strong> ${stats.lexical_diversity ?? 0}</p>
      <p><strong>Repetition Ratio:</strong> ${stats.repetition_ratio ?? 0}</p>
    `;

    const signals = result.signals || [];
    signalsBox.innerHTML = signals.length
      ? signals.map(s => `<li>${s}</li>`).join("")
      : "<li>No strong AI signals detected.</li>";

    const sentenceResults = result.sentence_results || [];
    sentenceResultsBox.innerHTML = sentenceResults.map((item, index) => {
      const cls = item.label === "Likely AI" ? "sentence ai"
        : item.label === "Mixed" ? "sentence mixed"
        : "sentence human";

      return `
        <div class="${cls}">
          <div class="sentence-head">
            <strong>Sentence ${index + 1}</strong>
            <span>${scoreToPercent(item.ai_score)}% AI</span>
          </div>
          <p>${item.sentence}</p>
        </div>
      `;
    }).join("");

    resultBox.classList.remove("hidden");
  } catch (err) {
    showError("Failed to analyze text");
  } finally {
    loading.classList.add("hidden");
  }
});