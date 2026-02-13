function render({ model, el }) {
  const container = document.createElement("div");
  container.className = "grpo-gdpo-root";
  el.appendChild(container);

  // Helper functions for statistics
  function mean(arr) {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  function std(arr) {
    if (arr.length <= 1) return 0;
    const m = mean(arr);
    // Use sample std (n-1) to match the paper
    const variance = arr.reduce((acc, val) => acc + (val - m) ** 2, 0) / (arr.length - 1);
    return Math.sqrt(variance);
  }

  function normalize(arr) {
    const m = mean(arr);
    const s = std(arr);
    if (s === 0) return arr.map(() => 0);
    return arr.map((v) => (v - m) / s);
  }

  // Calculate GRPO advantage (normalize total reward)
  function calcGrpoAdvantages(rewards) {
    const totals = rewards.map(
      (r) => r.correctness + r.style + r.conciseness
    );
    return normalize(totals);
  }

  // Calculate GDPO advantage (normalize each dimension, then sum)
  function calcGdpoAdvantages(rewards) {
    const correctness = rewards.map((r) => r.correctness);
    const style = rewards.map((r) => r.style);
    const conciseness = rewards.map((r) => r.conciseness);

    const normCorrectness = normalize(correctness);
    const normStyle = normalize(style);
    const normConciseness = normalize(conciseness);

    return rewards.map(
      (_, i) => normCorrectness[i] + normStyle[i] + normConciseness[i]
    );
  }

  function formatNumber(n) {
    if (n === 0) return "0.000";
    return n.toFixed(3);
  }

  function draw() {
    const rewards = model.get("rewards") || [];

    const grpoAdvantages = calcGrpoAdvantages(rewards);
    const gdpoAdvantages = calcGdpoAdvantages(rewards);

    container.innerHTML = "";

    // Create table
    const table = document.createElement("table");
    table.className = "grpo-gdpo-table";

    // Header row
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    const headers = [
      "",
      "Correctness",
      "Style",
      "Conciseness",
      "Total",
      "GRPO Adv",
      "GDPO Adv",
      "Difference",
    ];
    headers.forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body rows
    const tbody = document.createElement("tbody");
    rewards.forEach((reward, rowIndex) => {
      const row = document.createElement("tr");

      // Rollout label
      const labelCell = document.createElement("td");
      labelCell.className = "rollout-label";
      labelCell.textContent = `Rollout ${rowIndex}`;
      row.appendChild(labelCell);

      // Reward cells (clickable)
      ["correctness", "style", "conciseness"].forEach((dim) => {
        const cell = document.createElement("td");
        cell.className = "reward-cell";
        cell.dataset.value = reward[dim];
        cell.textContent = reward[dim];
        cell.addEventListener("click", () => {
          const newRewards = [...rewards];
          newRewards[rowIndex] = {
            ...newRewards[rowIndex],
            [dim]: reward[dim] === 1 ? 0 : 1,
          };
          model.set("rewards", newRewards);
          model.save_changes();
        });
        row.appendChild(cell);
      });

      // Total
      const total = reward.correctness + reward.style + reward.conciseness;
      const totalCell = document.createElement("td");
      totalCell.className = "computed-cell";
      totalCell.textContent = total;
      row.appendChild(totalCell);

      // GRPO Advantage
      const grpoCell = document.createElement("td");
      grpoCell.className = "computed-cell";
      grpoCell.textContent = formatNumber(grpoAdvantages[rowIndex]);
      row.appendChild(grpoCell);

      // GDPO Advantage
      const gdpoCell = document.createElement("td");
      gdpoCell.className = "computed-cell";
      gdpoCell.textContent = formatNumber(gdpoAdvantages[rowIndex]);
      row.appendChild(gdpoCell);

      // Difference
      const diff = gdpoAdvantages[rowIndex] - grpoAdvantages[rowIndex];
      const diffCell = document.createElement("td");
      diffCell.className = "diff-cell";
      if (Math.abs(diff) > 0.001) {
        diffCell.classList.add("has-diff");
      }
      diffCell.textContent = formatNumber(diff);
      row.appendChild(diffCell);

      tbody.appendChild(row);
    });
    table.appendChild(tbody);

    container.appendChild(table);

    // Add/Remove buttons
    const buttonRow = document.createElement("div");
    buttonRow.className = "button-row";

    const addBtn = document.createElement("button");
    addBtn.textContent = "+ Add Rollout";
    addBtn.className = "action-btn";
    addBtn.addEventListener("click", () => {
      const newRewards = [
        ...rewards,
        { correctness: 0, style: 0, conciseness: 0 },
      ];
      model.set("rewards", newRewards);
      model.save_changes();
    });
    buttonRow.appendChild(addBtn);

    const removeBtn = document.createElement("button");
    removeBtn.textContent = "- Remove Last";
    removeBtn.className = "action-btn";
    removeBtn.disabled = rewards.length <= 2;
    removeBtn.addEventListener("click", () => {
      if (rewards.length > 2) {
        const newRewards = rewards.slice(0, -1);
        model.set("rewards", newRewards);
        model.save_changes();
      }
    });
    buttonRow.appendChild(removeBtn);

    container.appendChild(buttonRow);
  }

  // Listen for changes
  model.on("change:rewards", draw);

  // Initial render
  draw();

  return () => {
    // Cleanup
  };
}

export default { render };
