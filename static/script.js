// ===== DOM Elements =====
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const resultsContainer = document.getElementById("resultsContainer");

// Hamburger Menu
const hamburgerToggle = document.getElementById("hamburgerToggle");
const hamburgerMenu = document.getElementById("hamburgerMenu");
const insertMenuItem = document.getElementById("insertMenuItem");
const dbSelectorMenuItem = document.getElementById("dbSelectorMenuItem");

// Insert Modal
const insertModal = document.getElementById("insertModal");
const insertText = document.getElementById("insertText");
const insertPayload = document.getElementById("insertPayload");
const insertSubmitBtn = document.getElementById("insertSubmitBtn");
const insertCancelBtn = document.getElementById("insertCancelBtn");
const insertMessage = document.getElementById("insertMessage");

// Database Modal
const dbModal = document.getElementById("dbModal");
const dbSelect = document.getElementById("dbSelect");
const dbLoadBtn = document.getElementById("dbLoadBtn");
const dbCancelBtn = document.getElementById("dbCancelBtn");
const dbMessage = document.getElementById("dbMessage");

// ===== Hamburger Menu Control =====
hamburgerToggle.addEventListener("click", () => {
  hamburgerMenu.classList.toggle("open");
});

// Close menu when clicking outside
document.addEventListener("click", (e) => {
  if (
    !hamburgerMenu.contains(e.target) &&
    !hamburgerToggle.contains(e.target)
  ) {
    hamburgerMenu.classList.remove("open");
  }
});

// ===== Insert Data Menu Item =====
insertMenuItem.addEventListener("click", () => {
  hamburgerMenu.classList.remove("open");
  insertModal.classList.add("open");
  insertText.focus();
});

insertCancelBtn.addEventListener("click", () => {
  insertModal.classList.remove("open");
  clearInsertForm();
});

insertSubmitBtn.addEventListener("click", submitInsert);

// ===== Database Selector Menu Item =====
dbSelectorMenuItem.addEventListener("click", () => {
  hamburgerMenu.classList.remove("open");
  dbModal.classList.add("open");
  loadAvailableDatabases();
});

dbCancelBtn.addEventListener("click", () => {
  dbModal.classList.remove("open");
});

dbLoadBtn.addEventListener("click", loadSelectedDatabase);

// Focus input after splash screen
setTimeout(() => {
  searchInput.focus();
}, 2000);

// Search on button click
searchBtn.addEventListener("click", performSearch);

// Search on Enter key
searchInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    performSearch();
  }
});

// ===== SEARCH FUNCTION =====

async function performSearch() {
  const query = searchInput.value.trim();

  if (!query) {
    resultsContainer.innerHTML =
      '<div class="error-message">Please enter a search query</div>';
    return;
  }

  // Show loading state
  searchBtn.classList.add("loading");
  searchBtn.disabled = true;
  resultsContainer.innerHTML =
    '<div><span class="loading-spinner"></span>Searching...</div>';

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Search failed");
    }

    displayResults(data.results);
  } catch (error) {
    resultsContainer.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
  } finally {
    searchBtn.classList.remove("loading");
    searchBtn.disabled = false;
  }
}

function displayResults(results) {
  if (results.length === 0) {
    resultsContainer.innerHTML =
      '<div class="no-results">No results found. Try a different query.</div>';
    return;
  }

  // Check if main result already exists
  let mainResultDiv = document.querySelector(".main-result");

  if (!mainResultDiv) {
    // First time - create the initial structure only
    let html =
      '<div class="main-result"><div class="main-answer" id="typing-answer"></div></div>';
    resultsContainer.innerHTML = html;
    mainResultDiv = document.querySelector(".main-result");
  }

  // Update the main answer with typing animation
  if (results.length > 0) {
    const mainResult = results[0];
    const answerElement = document.getElementById("typing-answer");
    const fullText = mainResult.answer || mainResult.text;

    // Type the new text (slower at 50ms per character)
    typeText(answerElement, fullText, 50);

    // Update or add main text (context)
    let mainTextDiv = mainResultDiv.querySelector(".main-text");
    if (mainResult.text) {
      if (!mainTextDiv) {
        mainTextDiv = document.createElement("div");
        mainTextDiv.className = "main-text";
        mainResultDiv.appendChild(mainTextDiv);
      }
      mainTextDiv.textContent = mainResult.text;
    }
  }

  // Update suggestions (next 2 results)
  const suggestions = results.slice(1, 3);
  let suggestionsContainer = document.querySelector(".suggestions-container");

  if (suggestions.length > 0) {
    if (!suggestionsContainer) {
      suggestionsContainer = document.createElement("div");
      suggestionsContainer.className = "suggestions-container";
      resultsContainer.appendChild(suggestionsContainer);
    }

    // Clear and rebuild suggestions
    suggestionsContainer.innerHTML = "";

    suggestions.forEach((result) => {
      const suggestionHtml = `
                        <div class="suggestion-item">
                            <div class="suggestion-header-text">Did you mean...</div>
                            <div class="suggestion-text">${escapeHtml(result.text)}</div>
                        </div>
                    `;
      suggestionsContainer.insertAdjacentHTML("beforeend", suggestionHtml);
    });
  } else if (suggestionsContainer) {
    // Remove suggestions if none available
    suggestionsContainer.remove();
  }
}

function typeText(element, text, speed) {
  let index = 0;
  element.textContent = "";

  function type() {
    if (index < text.length) {
      element.textContent += text[index];
      index++;
      setTimeout(type, speed);
    } else {
      // Done typing - ensure full text is displayed
      element.textContent = text;
    }
  }

  type();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ===== INSERT DATA FUNCTION =====
async function submitInsert() {
  const text = insertText.value.trim();

  if (!text) {
    showInsertMessage("❌ Text is required", true);
    return;
  }

  const payload = insertPayload.value.trim() || "";

  insertSubmitBtn.disabled = true;
  showInsertMessage("⏳ Inserting...", false);

  try {
    const response = await fetch("/api/insert", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, payload }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Insert failed");
    }

    showInsertMessage("✅ Data inserted successfully!", false);
    setTimeout(() => {
      insertModal.classList.remove("open");
      clearInsertForm();
    }, 1500);
  } catch (error) {
    showInsertMessage(`❌ Error: ${error.message}`, true);
  } finally {
    insertSubmitBtn.disabled = false;
  }
}

function showInsertMessage(msg, isError) {
  insertMessage.innerHTML = `<div class="error-message" style="color: ${isError ? "#ff6464" : "#4cff00"}; border-color: ${isError ? "rgba(255, 100, 100, 0.5)" : "rgba(76, 255, 0, 0.5)"}; background: ${isError ? "rgba(255, 50, 50, 0.1)" : "rgba(76, 255, 0, 0.1)"};">${msg}</div>`;
}

function clearInsertForm() {
  insertText.value = "";
  insertPayload.value = "";
  insertMessage.innerHTML = "";
  insertSubmitBtn.disabled = false;
}

// ===== DATABASE SELECTOR FUNCTION =====
async function loadAvailableDatabases() {
  try {
    const response = await fetch("/api/databases");
    const data = await response.json();

    dbSelect.innerHTML = "";
    if (data.databases && data.databases.length > 0) {
      data.databases.forEach((db) => {
        const option = document.createElement("option");
        option.value = db;
        option.textContent = db;
        if (db === data.current) {
          option.selected = true;
          option.textContent += " (current)";
        }
        dbSelect.appendChild(option);
      });
    } else {
      const option = document.createElement("option");
      option.textContent = "No databases found";
      option.disabled = true;
      dbSelect.appendChild(option);
    }
    dbMessage.innerHTML = "";
  } catch (error) {
    dbMessage.innerHTML = `<div class="error-message">❌ Error loading databases: ${error.message}</div>`;
  }
}

async function loadSelectedDatabase() {
  const dbName = dbSelect.value;

  if (!dbName) {
    dbMessage.innerHTML =
      '<div class="error-message">Please select a database</div>';
    return;
  }

  dbLoadBtn.disabled = true;
  dbMessage.innerHTML = '<div style="color: #00d4ff;">⏳ Loading...</div>';

  try {
    const response = await fetch("/api/load-database", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ database: dbName }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Load failed");
    }

    dbMessage.innerHTML = `<div class="error-message" style="color: #4cff00; border-color: rgba(76, 255, 0, 0.5); background: rgba(76, 255, 0, 0.1);">✅ Loaded ${dbName} (${data.vectors_loaded} vectors)</div>`;
    setTimeout(() => {
      dbModal.classList.remove("open");
    }, 1500);
  } catch (error) {
    dbMessage.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
  } finally {
    dbLoadBtn.disabled = false;
  }
}
