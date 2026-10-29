/**
 * Floramind Gardening Assistant - Frontend Chat Logic
 * Coordinates RAG chat turns, citation rendering, and UI interactions.
 */

document.addEventListener('DOMContentLoaded', () => {
  const chatMessages = document.getElementById('chatMessages');
  const chatForm = document.getElementById('chatForm');
  const messageInput = document.getElementById('messageInput');
  const sendButton = document.getElementById('sendButton');
  const typingIndicator = document.getElementById('typingIndicator');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const charCount = document.getElementById('charCount');
  const quickPromptsContainer = document.getElementById('quickPrompts');

  // Conversation history maintained in memory for multi-turn context
  let conversationHistory = [];

  // Auto-resize textarea
  messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 150)}px`;
    charCount.textContent = `${messageInput.value.length}/2000`;
  });

  // Handle Enter key for submit (Shift+Enter for newline)
  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  // Handle Quick Prompt Chips
  if (quickPromptsContainer) {
    quickPromptsContainer.addEventListener('click', (e) => {
      const chip = e.target.closest('.prompt-chip');
      if (chip && chip.dataset.prompt) {
        messageInput.value = chip.dataset.prompt;
        messageInput.style.height = 'auto';
        messageInput.style.height = `${Math.min(messageInput.scrollHeight, 150)}px`;
        charCount.textContent = `${messageInput.value.length}/2000`;
        chatForm.dispatchEvent(new Event('submit'));
      }
    });
  }

  // Clear Chat History
  clearChatBtn.addEventListener('click', () => {
    conversationHistory = [];
    location.reload();
  });

  // Form submission handler
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = messageInput.value.trim();
    if (!message) return;

    // Append User Message to UI
    appendUserMessage(message);

    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    charCount.textContent = '0/2000';
    messageInput.disabled = true;
    sendButton.disabled = true;

    // Show typing indicator
    showTypingIndicator(true);

    try {
      // Call Floramind RAG API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          history: conversationHistory.slice(-6), // Send recent turns
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate response.');
      }

      const data = await response.json();

      // Append Assistant Message to UI
      appendAssistantMessage(data);

      // Record in conversation history
      conversationHistory.push({ role: 'user', content: message });
      conversationHistory.push({ role: 'model', content: data.answer });

    } catch (err) {
      console.error('Chat error:', err);
      appendErrorMessage(err.message || 'An unexpected network error occurred.');
    } finally {
      showTypingIndicator(false);
      messageInput.disabled = false;
      sendButton.disabled = false;
      messageInput.focus();
    }
  });

  /**
   * Appends user message bubble to chat window.
   */
  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user-row';

    const bubbleWrapper = document.createElement('div');
    bubbleWrapper.className = 'message-bubble-wrapper';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;

    bubbleWrapper.appendChild(bubble);

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar shadow-sm';
    avatar.innerHTML = '<i class="fa-solid fa-user"></i>';

    row.appendChild(bubbleWrapper);
    row.appendChild(avatar);

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  /**
   * Appends Floramind assistant message bubble along with citations.
   */
  function appendAssistantMessage(data) {
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar shadow-sm';
    avatar.innerHTML = '<i class="fa-solid fa-seedling"></i>';

    const bubbleWrapper = document.createElement('div');
    bubbleWrapper.className = 'message-bubble-wrapper';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Grounding Status Pill
    const groundingBadge = document.createElement('div');
    if (data.is_grounded && data.citations && data.citations.length > 0) {
      groundingBadge.className = 'grounding-badge grounded shadow-xs';
      groundingBadge.innerHTML = `<i class="fa-solid fa-shield-halved"></i> Grounded with ${data.citations.length} Knowledge References`;
    } else {
      groundingBadge.className = 'grounding-badge insufficient shadow-xs';
      groundingBadge.innerHTML = `<i class="fa-solid fa-circle-info"></i> General Guidance &bull; Limited Knowledge Match`;
    }
    bubble.appendChild(groundingBadge);

    // Markdown rendered answer body
    const bodyContent = document.createElement('div');
    bodyContent.className = 'markdown-body';
    if (window.marked) {
      bodyContent.innerHTML = marked.parse(data.answer);
    } else {
      bodyContent.textContent = data.answer;
    }
    bubble.appendChild(bodyContent);

    // Citations Accordion / Section if available
    if (data.citations && data.citations.length > 0) {
      const citationsWrapper = document.createElement('div');
      citationsWrapper.className = 'citations-wrapper';

      const header = document.createElement('div');
      header.className = 'citations-header';
      header.innerHTML = `<i class="fa-solid fa-book-bookmark text-emerald"></i> Verified Knowledge Citations (${data.citations.length}):`;
      citationsWrapper.appendChild(header);

      data.citations.forEach((cit, idx) => {
        const card = document.createElement('div');
        card.className = 'citation-card shadow-xs';

        const categoryTag = cit.category ? `<span class="badge bg-light text-secondary me-1 border">${cit.category}</span>` : '';
        const sectionTag = cit.section ? `<span class="text-muted fs-xxs">&bull; ${cit.section}</span>` : '';
        const scorePercent = Math.round(cit.score * 100);

        card.innerHTML = `
          <div class="citation-meta" data-bs-toggle="collapse" data-bs-target="#cit-collapse-${idx}-${Date.now()}">
            <div>
              ${categoryTag}
              <span>${cit.title}</span>
              ${sectionTag}
            </div>
            <span class="citation-score-badge" title="Vector Cosine Similarity: ${cit.score}">
              ${scorePercent}% match
            </span>
          </div>
          <div class="collapse show" id="cit-collapse-${idx}-${Date.now()}">
            <div class="citation-snippet">
              "${cit.snippet}"
            </div>
          </div>
        `;
        citationsWrapper.appendChild(card);
      });

      bubble.appendChild(citationsWrapper);
    }

    bubbleWrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(bubbleWrapper);

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  /**
   * Appends error alert inside chat window.
   */
  function appendErrorMessage(errText) {
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar bg-danger text-white shadow-sm';
    avatar.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';

    const bubbleWrapper = document.createElement('div');
    bubbleWrapper.className = 'message-bubble-wrapper';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble border-danger-subtle bg-danger-subtle text-danger';
    bubble.innerHTML = `<strong>Error:</strong> ${errText}`;

    bubbleWrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(bubbleWrapper);

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function showTypingIndicator(show) {
    if (show) {
      typingIndicator.classList.remove('d-none');
    } else {
      typingIndicator.classList.add('d-none');
    }
    scrollToBottom();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
});
