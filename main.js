/**
 * K. Yamuna — Portfolio Main UI & Interaction Controller
 * 3D Card Tilt • Custom Cursor • Interactive Terminal • Toast Notifications • Copy/Print
 */

document.addEventListener('DOMContentLoaded', () => {
  'use strict';

  // --- 1. Custom Interactive Cursor Follower ---
  const cursorDot = document.getElementById('cursorDot');
  const cursorGlow = document.getElementById('cursorGlow');

  if (cursorDot && cursorGlow && window.matchMedia('(pointer: fine)').matches) {
    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;
    let dotX = mouseX;
    let dotY = mouseY;
    let glowX = mouseX;
    let glowY = mouseY;

    window.addEventListener('mousemove', (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
      cursorDot.style.left = `${mouseX}px`;
      cursorDot.style.top = `${mouseY}px`;
    }, { passive: true });

    function renderCursor() {
      // Lerp smooth trailing for glow ring
      glowX += (mouseX - glowX) * 0.15;
      glowY += (mouseY - glowY) * 0.15;
      cursorGlow.style.left = `${glowX}px`;
      cursorGlow.style.top = `${glowY}px`;
      requestAnimationFrame(renderCursor);
    }
    renderCursor();

    // Scale cursor when hovering over interactive elements
    const interactiveElements = document.querySelectorAll('a, button, input, textarea, .tilt-card, .btn-copy-chip');
    interactiveElements.forEach((el) => {
      el.addEventListener('mouseenter', () => {
        cursorGlow.style.width = '54px';
        cursorGlow.style.height = '54px';
        cursorGlow.style.borderColor = 'rgba(139, 92, 246, 0.7)';
        cursorGlow.style.background = 'rgba(139, 92, 246, 0.1)';
      });
      el.addEventListener('mouseleave', () => {
        cursorGlow.style.width = '36px';
        cursorGlow.style.height = '36px';
        cursorGlow.style.borderColor = 'rgba(6, 182, 212, 0.45)';
        cursorGlow.style.background = 'rgba(6, 182, 212, 0.05)';
      });
    });
  }

  // --- 2. 3D Card Tilt Effect ---
  const tiltCards = document.querySelectorAll('.tilt-card');
  tiltCards.forEach((card) => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      const rotateX = ((y - centerY) / centerY) * -9; // Max 9 deg tilt
      const rotateY = ((x - centerX) / centerX) * 9;

      card.style.transform = `perspective(1000px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) scale3d(1.02, 1.02, 1.02)`;
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
    });
  });

  // --- 3. Navbar Sticky Style & Mobile Menu Toggle ---
  const navbar = document.getElementById('navbar');
  const mobileToggle = document.getElementById('mobileToggle');
  const navMenu = document.getElementById('navMenu');

  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  }, { passive: true });

  if (mobileToggle && navMenu) {
    mobileToggle.addEventListener('click', () => {
      navMenu.classList.toggle('open');
      const isOpen = navMenu.classList.contains('open');
      mobileToggle.setAttribute('aria-expanded', isOpen);
    });

    // Close menu when clicking link
    navMenu.querySelectorAll('.nav-link').forEach((link) => {
      link.addEventListener('click', () => {
        navMenu.classList.remove('open');
      });
    });
  }

  // --- 4. Scroll-Spy Navigation Active Link Detection ---
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  function updateActiveNav() {
    let current = '';
    const scrollPosition = window.pageYOffset + 160;

    sections.forEach((section) => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.offsetHeight;
      if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach((link) => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  }

  window.addEventListener('scroll', updateActiveNav, { passive: true });
  updateActiveNav();

  // --- 5. Toast Notification System ---
  const toastContainer = document.getElementById('toastContainer');

  function showToast(message, icon = 'fa-check-circle') {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<i class="fa-solid ${icon}"></i><span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 3600);
  }

  // --- 6. One-Click Copy-to-Clipboard ---
  // Copy Email Buttons
  const copyEmailButtons = document.querySelectorAll('.copy-email-btn');
  copyEmailButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const email = btn.getAttribute('data-email') || 'yamunakanagaraj9@gmail.com';
      navigator.clipboard.writeText(email).then(() => {
        showToast(`Email copied: ${email}`, 'fa-envelope');
      }).catch(() => {
        showToast(`Email: ${email}`, 'fa-envelope');
      });
    });
  });

  // Copy Phone Buttons
  const copyPhoneButtons = document.querySelectorAll('.copy-phone-btn');
  copyPhoneButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const phone = btn.getAttribute('data-phone') || '7010645757';
      navigator.clipboard.writeText(phone).then(() => {
        showToast(`Phone number copied: ${phone}`, 'fa-phone');
      }).catch(() => {
        showToast(`Phone: ${phone}`, 'fa-phone');
      });
    });
  });

  // --- 7. Print Resume Trigger ---
  const printResumeBtn = document.getElementById('printResumeBtn');
  const triggerPrintBtn = document.getElementById('triggerPrintBtn');

  function triggerPrint() {
    showToast('Opening clean print / PDF view...', 'fa-file-pdf');
    setTimeout(() => {
      window.print();
    }, 400);
  }

  if (printResumeBtn) printResumeBtn.addEventListener('click', triggerPrint);
  if (triggerPrintBtn) triggerPrintBtn.addEventListener('click', triggerPrint);

  // --- 8. Interactive Live Skill Playground ---
  const runCodeBtn = document.getElementById('runCodeBtn');
  const resetPlaygroundBtn = document.getElementById('resetPlaygroundBtn');
  const terminalOutput = document.getElementById('terminalOutput');

  if (runCodeBtn && terminalOutput) {
    runCodeBtn.addEventListener('click', () => {
      terminalOutput.innerHTML = '';
      const lines = [
        { text: '> [1/4] Compiling candidate profile script...', class: 'text-muted', delay: 100 },
        { text: '> [2/4] Validating academic credentials: Muthayammal College of Arts and Science', class: 'text-violet', delay: 400 },
        { text: '> [3/4] B.Sc Computer Science (Third Year) score verified: 75%', class: 'text-cyan', delay: 700 },
        { text: '✓ Candidate Record: K. Yamuna', class: 'text-emerald', delay: 1000 },
        { text: '✓ Status: Motivated B.Sc Computer Science Student ready for professional roles!', class: 'text-primary', delay: 1300 },
        { text: '>> Execution finished successfully with code 0.', class: 'text-muted', delay: 1600 }
      ];

      lines.forEach((item) => {
        setTimeout(() => {
          const div = document.createElement('div');
          div.className = `terminal-line ${item.class}`;
          div.textContent = item.text;
          terminalOutput.appendChild(div);
          terminalOutput.scrollTop = terminalOutput.scrollHeight;
        }, item.delay);
      });

      showToast('Simulation executed successfully!', 'fa-terminal');
    });
  }

  if (resetPlaygroundBtn && terminalOutput) {
    resetPlaygroundBtn.addEventListener('click', () => {
      terminalOutput.innerHTML = `
        <div class="terminal-line text-muted">> Ready to execute... click "Run Code Simulation"</div>
        <div class="terminal-line text-cyan">> Environment: B.Sc Computer Science Runtime v2026</div>
      `;
      showToast('Console reset.', 'fa-rotate-right');
    });
  }

  // --- 9. Contact Form Handling ---
  const contactForm = document.getElementById('portfolioContactForm');
  if (contactForm) {
    contactForm.addEventListener('submit', (e) => {
      e.preventDefault();

      const name = document.getElementById('senderName').value.trim();
      const email = document.getElementById('senderEmail').value.trim();
      const subject = document.getElementById('messageSubject').value.trim() || 'Opportunity for B.Sc CS Candidate';
      const body = document.getElementById('messageBody').value.trim();

      if (!name || !email || !body) {
        showToast('Please fill in all required fields.', 'fa-triangle-exclamation');
        return;
      }

      // Format email body for mailto
      const formattedBody = `From: ${name} (${email})\n\nMessage:\n${body}\n\n---\nSent via K. Yamuna Portfolio Website`;
      const mailtoUrl = `mailto:yamunakanagaraj9@gmail.com?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(formattedBody)}`;

      showToast('Opening your email client to send message...', 'fa-paper-plane');

      setTimeout(() => {
        window.location.href = mailtoUrl;
      }, 600);
    });
  }

  // --- 10. Copy Git Clone Command ---
  const copyCloneButtons = document.querySelectorAll('.copy-clone-btn');
  copyCloneButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const cmd = btn.getAttribute('data-clone') || 'git clone https://github.com/24ucs168-dev/airline-ml-project.git';
      navigator.clipboard.writeText(cmd).then(() => {
        showToast('Repository clone command copied!', 'fa-terminal');
      }).catch(() => {
        showToast(`Command: ${cmd}`, 'fa-terminal');
      });
    });
  });

  // --- 11. Project Tab Navigation Switcher ---
  const projectTabButtons = document.querySelectorAll('.project-tab-btn');
  const projectTabPanels = document.querySelectorAll('.project-tab-panel');

  projectTabButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      projectTabButtons.forEach((b) => b.classList.remove('active'));
      projectTabPanels.forEach((p) => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });

  // --- 12. SkyPulse Customer Satisfaction Simulator Controller ---
  const simTravelClass = document.getElementById('simTravelClass');
  const simTravelType = document.getElementById('simTravelType');
  const simCustomerType = document.getElementById('simCustomerType');

  const sliderEntertainment = document.getElementById('sliderEntertainment');
  const sliderSeat = document.getElementById('sliderSeat');
  const sliderOnline = document.getElementById('sliderOnline');
  const sliderFood = document.getElementById('sliderFood');
  const sliderDistance = document.getElementById('sliderDistance');
  const sliderDelay = document.getElementById('sliderDelay');

  const valEntertainment = document.getElementById('valEntertainment');
  const valSeat = document.getElementById('valSeat');
  const valOnline = document.getElementById('valOnline');
  const valFood = document.getElementById('valFood');
  const valDistance = document.getElementById('valDistance');
  const valDelay = document.getElementById('valDelay');

  const resultStatusIcon = document.getElementById('resultStatusIcon');
  const resultVerdict = document.getElementById('resultVerdict');
  const confPercentage = document.getElementById('confPercentage');
  const confBarFill = document.getElementById('confBarFill');
  const probSatisfied = document.getElementById('probSatisfied');
  const probDissatisfied = document.getElementById('probDissatisfied');
  const inferenceNote = document.getElementById('inferenceNote');

  const presetHappy = document.getElementById('presetHappy');
  const presetFrustrated = document.getElementById('presetFrustrated');
  const presetRandom = document.getElementById('presetRandom');

  function updateSimulator() {
    if (!sliderEntertainment || !resultVerdict) return;

    const entertainment = parseInt(sliderEntertainment.value, 10);
    const seat = parseInt(sliderSeat.value, 10);
    const online = parseInt(sliderOnline.value, 10);
    const food = parseInt(sliderFood.value, 10);
    const distance = parseInt(sliderDistance.value, 10);
    const delay = parseInt(sliderDelay.value, 10);

    const travelClass = simTravelClass ? simTravelClass.value : 'Business';
    const travelType = simTravelType ? simTravelType.value : 'Business travel';
    const customerType = simCustomerType ? simCustomerType.value : 'Loyal Customer';

    // Update Display Values
    if (valEntertainment) valEntertainment.textContent = `${entertainment} / 5`;
    if (valSeat) valSeat.textContent = `${seat} / 5`;
    if (valOnline) valOnline.textContent = `${online} / 5`;
    if (valFood) valFood.textContent = `${food} / 5`;
    if (valDistance) valDistance.textContent = `${distance.toLocaleString()} mi`;
    if (valDelay) valDelay.textContent = `${delay} min`;

    // SkyPulse Random Forest Model Simulation Logic (derived from top Gini importances)
    let score = 0.50;

    // Inflight entertainment (25.4% weight)
    score += (entertainment - 2.5) * 0.088;
    // Seat comfort (21.8% weight)
    score += (seat - 2.5) * 0.076;
    // Online booking & support (17.6% + 14.9% combined)
    score += (online - 2.5) * 0.062;
    // Food and drink (11.2% weight)
    score += (food - 2.5) * 0.038;

    // Class modifiers
    if (travelClass === 'Business') score += 0.15;
    else if (travelClass === 'Eco') score -= 0.12;
    else if (travelClass === 'Eco Plus') score += 0.02;

    // Travel type modifiers
    if (travelType === 'Business travel') score += 0.07;
    else score -= 0.07;

    // Customer loyalty modifiers
    if (customerType === 'Loyal Customer') score += 0.06;
    else score -= 0.06;

    // Delay penalty: exponential sensitivity above 30 min
    if (delay > 0) {
      const delayPenalty = Math.min(0.40, (delay / 15) * 0.03);
      score -= delayPenalty;
    }

    // Distance effect: long flights with good comfort boost satisfaction; bad comfort hurts more
    if (distance > 2000) {
      if (seat >= 4) score += 0.04;
      else if (seat <= 2) score -= 0.05;
    }

    // Bound probability between 2.1% and 98.4%
    score = Math.max(0.021, Math.min(0.984, score));

    const satPct = (score * 100).toFixed(1);
    const dissatPct = ((1 - score) * 100).toFixed(1);

    if (score >= 0.50) {
      resultVerdict.textContent = 'Satisfied';
      resultVerdict.className = 'result-verdict text-emerald';
      resultStatusIcon.className = 'result-icon satisfied';
      resultStatusIcon.innerHTML = '<i class="fa-solid fa-face-smile"></i>';
      confPercentage.textContent = `${satPct}%`;
      confBarFill.className = 'conf-bar-fill fill-satisfied';
      confBarFill.style.width = `${satPct}%`;
    } else {
      resultVerdict.textContent = 'Dissatisfied';
      resultVerdict.className = 'result-verdict text-rose';
      resultStatusIcon.className = 'result-icon dissatisfied';
      resultStatusIcon.innerHTML = '<i class="fa-solid fa-face-frown"></i>';
      confPercentage.textContent = `${dissatPct}%`;
      confBarFill.className = 'conf-bar-fill fill-dissatisfied';
      confBarFill.style.width = `${dissatPct}%`;
    }

    if (probSatisfied) probSatisfied.textContent = `${satPct}%`;
    if (probDissatisfied) probDissatisfied.textContent = `${dissatPct}%`;

    // Dynamic contextual explanation
    if (inferenceNote) {
      if (delay > 45 && score < 0.5) {
        inferenceNote.textContent = `Extended flight delay of ${delay} minutes combined with low service ratings led to a high probability of passenger dissatisfaction.`;
      } else if (entertainment >= 4 && seat >= 4 && score >= 0.8) {
        inferenceNote.textContent = `High ratings for Inflight Entertainment (${entertainment}/5) and Seat Comfort (${seat}/5) were dominant factors driving ${satPct}% satisfaction.`;
      } else if (travelClass === 'Business' && score >= 0.7) {
        inferenceNote.textContent = `Business Class amenities and loyal customer retention heavily cushioned minor flight friction, resulting in a satisfied verdict.`;
      } else if (score < 0.5) {
        inferenceNote.textContent = `Below-average inflight comfort and economy travel friction resulted in a predicted dissatisfied passenger profile.`;
      } else {
        inferenceNote.textContent = `Balanced passenger ratings across digital services and physical cabin amenities produced a moderately satisfied classification.`;
      }
    }
  }

  // Attach event listeners to simulator controls
  const simControls = [
    simTravelClass, simTravelType, simCustomerType,
    sliderEntertainment, sliderSeat, sliderOnline, sliderFood, sliderDistance, sliderDelay
  ];
  simControls.forEach((ctrl) => {
    if (ctrl) {
      ctrl.addEventListener('input', () => {
        // Clear active preset highlight on manual tweak
        [presetHappy, presetFrustrated, presetRandom].forEach((p) => p && p.classList.remove('active'));
        updateSimulator();
      });
    }
  });

  // Presets Handlers
  if (presetHappy) {
    presetHappy.addEventListener('click', () => {
      [presetHappy, presetFrustrated, presetRandom].forEach((p) => p && p.classList.remove('active'));
      presetHappy.classList.add('active');

      if (simTravelClass) simTravelClass.value = 'Business';
      if (simTravelType) simTravelType.value = 'Business travel';
      if (simCustomerType) simCustomerType.value = 'Loyal Customer';

      if (sliderEntertainment) sliderEntertainment.value = 5;
      if (sliderSeat) sliderSeat.value = 5;
      if (sliderOnline) sliderOnline.value = 5;
      if (sliderFood) sliderFood.value = 4;
      if (sliderDistance) sliderDistance.value = 2150;
      if (sliderDelay) sliderDelay.value = 5;

      updateSimulator();
      showToast('Loaded preset: Happy Business Traveler', 'fa-star');
    });
  }

  if (presetFrustrated) {
    presetFrustrated.addEventListener('click', () => {
      [presetHappy, presetFrustrated, presetRandom].forEach((p) => p && p.classList.remove('active'));
      presetFrustrated.classList.add('active');

      if (simTravelClass) simTravelClass.value = 'Eco';
      if (simTravelType) simTravelType.value = 'Personal Travel';
      if (simCustomerType) simCustomerType.value = 'disloyal Customer';

      if (sliderEntertainment) sliderEntertainment.value = 2;
      if (sliderSeat) sliderSeat.value = 1;
      if (sliderOnline) sliderOnline.value = 2;
      if (sliderFood) sliderFood.value = 2;
      if (sliderDistance) sliderDistance.value = 450;
      if (sliderDelay) sliderDelay.value = 85;

      updateSimulator();
      showToast('Loaded preset: Frustrated Economy Traveler', 'fa-triangle-exclamation');
    });
  }

  if (presetRandom) {
    presetRandom.addEventListener('click', () => {
      [presetHappy, presetFrustrated, presetRandom].forEach((p) => p && p.classList.remove('active'));
      presetRandom.classList.add('active');

      const classes = ['Business', 'Eco', 'Eco Plus'];
      const types = ['Business travel', 'Personal Travel'];
      const customers = ['Loyal Customer', 'disloyal Customer'];

      if (simTravelClass) simTravelClass.value = classes[Math.floor(Math.random() * classes.length)];
      if (simTravelType) simTravelType.value = types[Math.floor(Math.random() * types.length)];
      if (simCustomerType) simCustomerType.value = customers[Math.floor(Math.random() * customers.length)];

      if (sliderEntertainment) sliderEntertainment.value = Math.floor(Math.random() * 6);
      if (sliderSeat) sliderSeat.value = Math.floor(Math.random() * 6);
      if (sliderOnline) sliderOnline.value = Math.floor(Math.random() * 6);
      if (sliderFood) sliderFood.value = Math.floor(Math.random() * 6);
      if (sliderDistance) sliderDistance.value = Math.floor(Math.random() * 40 + 2) * 100;
      if (sliderDelay) sliderDelay.value = Math.random() > 0.4 ? Math.floor(Math.random() * 12) * 10 : 0;

      updateSimulator();
      showToast('Generated random passenger scenario!', 'fa-shuffle');
    });
  }

  // Initial call for simulator
  if (sliderEntertainment) {
    updateSimulator();
  }
});
