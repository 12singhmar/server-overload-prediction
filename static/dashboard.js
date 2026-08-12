const form = document.querySelector("#simulator");
const title = document.querySelector("#result-title");
const copy = document.querySelector("#result-copy");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(form));
  Object.keys(data).forEach((key) => { data[key] = Number(data[key]); });
  title.textContent = "Running simulation…";
  copy.textContent = "";
  try {
    const response = await fetch("/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail?.message || "The simulation could not be completed.");
    title.textContent = `${(result.overload_probability * 100).toFixed(1)}% estimated overload probability`;
    copy.textContent = result.early_warning ? "Early-warning threshold reached. This is a synthetic demonstration result." : "Below the demonstration early-warning threshold. This is a synthetic demonstration result.";
  } catch (error) { title.textContent = "Simulation unavailable"; copy.textContent = error.message; }
});
