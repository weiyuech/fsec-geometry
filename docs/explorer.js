(() => {
  "use strict";
  const data = window.FSEC_CONTOUR;
  if (!data || !data.raySamples || data.raySamples.length !== 361) {
    const error = document.querySelector("#load-error");
    error.textContent = "The measured contour could not be loaded. Reload the page to try again.";
    error.hidden = false;
    document.querySelector(".experiment").hidden = true;
    document.querySelector(".under-plot").hidden = true;
    return;
  }

  const svg = document.querySelector("#graph");
  const description = document.querySelector("#graph-desc");
  const circle = document.querySelector("#circle");
  const mix = document.querySelector("#mix");
  document.querySelector("#threshold-value").textContent = data.threshold.toFixed(1);
  const angleOutput = document.querySelector("#angle");
  const distance = document.querySelector("#distance-number");
  const components = document.querySelector("#components");
  const accessibleResult = document.querySelector("#accessible-result");
  const comparisonAngle = document.querySelector("#comparison-angle");
  const rows = document.querySelector("#comparison-rows");
  const quadrants = [
    { key: "topRight", french: 1, tense: 1 },
    { key: "topLeft", french: -1, tense: 1 },
    { key: "bottomRight", french: 1, tense: -1 },
    { key: "bottomLeft", french: -1, tense: -1 }
  ];
  for (const q of quadrants) {
    const row = rows.insertRow();
    row.insertCell().textContent = q.french > 0 ? "Toward French" : "Reverse French";
    row.insertCell().textContent = q.tense > 0 ? "Future" : "Past";
    q.valueCell = row.insertCell();
  }

  function sample(q, mixAngle) {
    // The slider is in normalized display coordinates. Each exported ray is
    // the first intersection with a line segment of the measured contour.
    const direction = q.french > 0
      ? (q.tense > 0 ? mixAngle : 360 - mixAngle)
      : (q.tense > 0 ? 180 - mixAngle : 180 + mixAngle);
    return data.raySamples[direction];
  }

  const namespace = "http://www.w3.org/2000/svg";
  function node(parent, name, attrs, text = "") {
    const element = document.createElementNS(namespace, name);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, String(value));
    if (text) element.textContent = text;
    parent.appendChild(element);
    return element;
  }

  const picture = node(svg, "g", {"aria-hidden": "true"});
  function drawChart(selected, hit) {
    // Drawing in CSS pixels keeps graph text readable on narrow screens.
    const width = Math.round(svg.clientWidth);
    if (!width) return;
    const narrow = width < 405;
    const radius = Math.min(173, (width - (narrow ? 42 : 66)) / 2.74);
    const extent = 1.31 * radius, centerX = width / 2;
    const centerY = extent + 35, height = Math.ceil(centerY + extent + 34);
    const X = x => centerX + x * radius;
    const Y = y => centerY - y * radius;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    picture.replaceChildren();
    node(picture, "rect", {x:centerX, y:centerY-extent, width:extent, height:extent, fill:"#edf5f2"});
    for (const v of [-1, -.5, .5, 1]) {
      node(picture, "line", {class:"grid", x1:X(v), x2:X(v), y1:centerY-extent, y2:centerY+extent});
      node(picture, "line", {class:"grid", x1:centerX-extent, x2:centerX+extent, y1:Y(v), y2:Y(v)});
    }
    node(picture, "line", {class:"axis", x1:centerX-extent, x2:centerX+extent, y1:centerY, y2:centerY});
    node(picture, "line", {class:"axis", x1:centerX, x2:centerX, y1:centerY-extent, y2:centerY+extent});
    if (circle.checked) node(picture, "circle", {class:"circle-reference", cx:centerX, cy:centerY, r:radius});
    for (const q of quadrants) {
      const points = data.quadrants[q.key].points;
      const d = points.map(([x,y],i) => `${i === 0 ? "M" : "L"}${X(x)} ${Y(y)}`).join(" ");
      node(picture, "path", {class: q.key === selected.key ? "arc active" : "arc", d});
    }
    const x = X(hit.x), y = Y(hit.y);
    if (Math.abs(hit.x) > .00001 && Math.abs(hit.y) > .00001) {
      node(picture, "path", {class:"component", d:`M${centerX} ${centerY} H${x} V${y}`});
    }
    const dx = x - centerX, dy = y - centerY;
    const r = Math.hypot(dx,dy), ux = dx / r, uy = dy / r;
    const arrowWidth = narrow ? 4 : 5, arrowLength = narrow ? 10 : 13;
    const bx = x - arrowLength * ux, by = y - arrowLength * uy;
    node(picture, "line", {class:"push", x1:centerX, y1:centerY, x2:bx, y2:by});
    node(picture, "path", {fill:"var(--rust)", d:`M${x} ${y} L${bx-uy*arrowWidth} ${by+ux*arrowWidth} L${bx+uy*arrowWidth} ${by-ux*arrowWidth} Z`});
    node(picture, "circle", {cx:centerX, cy:centerY, r:3.2, fill:"var(--ink)"});
    node(picture, "text", {class:"origin-label", x:centerX+(hit.x>=0 ? -8 : 8),
      y:centerY+(hit.y>=0 ? 14 : -8),"text-anchor":hit.x>=0 ? "end" : "start"}, "No edit");
    const axisLabel = (text,x,y,anchor="middle") => node(picture,"text",
      {class:"axis-word",x,y,"text-anchor":anchor},text);
    axisLabel("Future ↑", centerX,centerY-extent-12);
    axisLabel("Past ↓", centerX,centerY+extent+23);
    // Name both French directions outside the data rectangle. Keeping the
    // labels on a separate baseline prevents them from covering a crossing.
    axisLabel(narrow ? "← Reverse" : "← Reverse French", centerX-extent,centerY+extent+23,"start");
    axisLabel(narrow ? "French →" : "Toward French →", centerX+extent,centerY+extent+23,"end");
    // Put the paper annotation inside the shaded quarter, well above the data.
    node(picture,"text",{class:"original-label",x:centerX+extent-4,
      y:centerY-extent+15,"text-anchor":"end"}, narrow ? "Paper’s sweep ++" : "Paper’s original sweep ++");
    for(const value of [-1,1]) {
      node(picture,"text",{class:"tick",x:X(value),y:centerY+17,"text-anchor":"middle"},value>0?"1":"−1");
      node(picture,"text",{class:"tick",x:centerX-9,y:Y(value)+4,"text-anchor":"end"},value>0?"1":"−1");
    }
  }

  function signed(value) {
    if (Math.abs(value) < 0.0005) return "0.000";
    return (value < 0 ? "−" : "+") + Math.abs(value).toFixed(3);
  }
  function current() {
    const french = Number(document.querySelector('input[name="french"]:checked').value);
    const tense = Number(document.querySelector('input[name="tense"]:checked').value);
    return quadrants.find(q => q.french === french && q.tense === tense);
  }
  function render(announce = false) {
    const q = current(), a = Number(mix.value);
    const hit = sample(q,a);
    angleOutput.value = `${a}°`;
    mix.setAttribute("aria-valuetext", a === 0 ? "0 degrees, French only" :
      a === 90 ? "90 degrees, tense only" : `${a} degrees${a === 45 ? ", equal components" : ""}`);
    distance.textContent = hit.radius.toFixed(3);
    components.textContent = `French component ${signed(hit.x)} · Tense component ${signed(hit.y)}`;
    comparisonAngle.textContent = `${a}°`;
    for (const other of quadrants) {
      other.valueCell.textContent = sample(other,a).radius.toFixed(3);
      other.valueCell.parentElement.setAttribute("aria-current", String(q.key === other.key));
    }
    const text = `${q.french > 0 ? "Toward French" : "Reverse French"}, ${q.tense > 0 ? "future" : "past"}, mixing angle ${a} degrees. The push reaches the boundary at ${hit.radius.toFixed(3)} axis units.`;
    description.textContent = text + " The upper-right quadrant is the original positive-positive sweep. The center is the unedited activation.";
    if (announce) accessibleResult.textContent = text;
    drawChart(q,hit);
  }
  for(const input of document.querySelectorAll('input[type="radio"]')) {
    input.addEventListener("change",() => render(true));
  }
  mix.addEventListener("input",() => render());
  mix.addEventListener("change",() => render(true));
  circle.addEventListener("change",() => render());
  let previousWidth = -1;
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      if (svg.clientWidth !== previousWidth) {
        previousWidth = svg.clientWidth;
        render();
      }
    }).observe(svg);
  } else {
    window.addEventListener("resize",() => render());
  }
  render();
})();
