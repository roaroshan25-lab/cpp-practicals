// Navigate between website sections
function showPage(pageId) {
  document.querySelectorAll(".page").forEach(page => {
    page.hidden = true;
  });

  document.getElementById(pageId).hidden = false;
}

// Search stock data through our secure backend
async function getStock(inputId) {
  const symbol = document.getElementById(inputId).value
    .trim()
    .toUpperCase();

  const resultId = inputId === "scannerSymbol"
    ? "scannerResult"
    : "compareResult";

  const result = document.getElementById(resultId);

  if (!symbol) {
    result.textContent = "Please enter a stock symbol.";
    return;
  }

  result.textContent = "Loading stock data...";

  try {
    const response = await fetch(
      `/api/stocks?symbol=${encodeURIComponent(symbol)}`
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Unable to load stock data.");
    }

    if (!data.values || data.values.length === 0) {
      throw new Error("No stock data found.");
    }

    const latest = data.values[0];

    result.innerHTML = `
      <h3>${symbol}</h3>
      <p>Latest available date: ${latest.datetime}</p>
      <p>Open: ${latest.open}</p>
      <p>High: ${latest.high}</p>
      <p>Low: ${latest.low}</p>
      <p>Close: ${latest.close}</p>
      <p>Volume: ${latest.volume}</p>
    `;

  } catch (error) {
    result.textContent = error.message;
  }
}
