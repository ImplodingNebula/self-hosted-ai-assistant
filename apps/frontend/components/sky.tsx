"use client";

import { useEffect, useRef } from "react";

import { useTheme } from "@/components/theme-provider";
import {
  AMBER,
  approach,
  cssRgb,
  ERROR_WASH,
  FIELD_TALL,
  fieldY,
  frameDue,
  glslVec3,
  MAGENTA,
  NIGHT,
  parallaxOffset,
  pickSkyRenderer,
  shaderFrameInterval,
  shaderSize,
  STAR_WHITE,
  STARS_FRAME_INTERVAL,
  type SkyRenderer,
  type SkyWeather,
  TEAL,
  tripletRgb,
  VIOLET,
  weatherFor,
} from "@/lib/sky";
import {
  observeTurnState,
  observeView,
  readTurnState,
  readView,
  SKY_HANDOFF_EVENT,
  SKY_JUMP_EVENT,
} from "@/lib/turn-state";

/**
 * Aurora's sky: a soft curtain of light behind the app, held back from the
 * reading column by the veil (themes/aurora.css), answering the turn with an
 * energy and a tint per state. Two layers: the sky itself, drawn by one WebGL
 * plane at half resolution (or by three CSS curtains where WebGL is missing),
 * and the constellations, a sharp 2D canvas of stars among which a spark now
 * and then traces a line through a few neighbours.
 *
 * Mounted once in the root layout; renders nothing unless aurora is the
 * active theme, and clears `data-sky` from the root when it is not, so other
 * themes carry no trace of it.
 */
export function Sky() {
  const { theme } = useTheme();
  const aurora = theme === "aurora";
  useEffect(() => {
    if (!aurora) delete document.documentElement.dataset.sky;
  }, [aurora]);
  return aurora ? <SkyCanvas /> : null;
}

/* The weather as the renderer holds it between frames (unit RGB). */
interface Weather {
  e: number;
  err: number;
  pulse: number;
  tint: [number, number, number];
}

function SkyCanvas() {
  const rootRef = useRef<HTMLDivElement>(null);
  const glRef = useRef<HTMLCanvasElement>(null);
  const starsRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const host = rootRef.current;
    const glCanvas = glRef.current;
    const starCanvas = starsRef.current;
    if (!host || !glCanvas || !starCanvas) return;
    const root = document.documentElement;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const reduceTransparency = window.matchMedia("(prefers-reduced-transparency: reduce)");

    /* ---------- the weather: eased toward the state's target ---------- */
    const first = weatherFor(readTurnState());
    const cur: Weather = { e: first.e, err: first.err ?? 0, pulse: 0, tint: [...first.tint] };
    let tgt: SkyWeather = first;
    const settled = () => Math.abs(cur.e - tgt.e) < 0.01 && cur.pulse < 0.02;
    const snap = () => {
      cur.e = tgt.e;
      cur.err = tgt.err ?? 0;
      cur.tint = [...tgt.tint];
      cur.pulse = 0;
    };
    const stepWeather = (dt: number) => {
      cur.e = approach(cur.e, tgt.e, dt);
      cur.err = approach(cur.err, tgt.err ?? 0, dt);
      cur.pulse *= Math.exp(-dt / 700);
      for (let i = 0; i < 3; i++) cur.tint[i] = approach(cur.tint[i], tgt.tint[i], dt);
    };
    /* What the sky shows this frame: eased while the shader runs, else the target. */
    const current = (): Weather =>
      shaderRunning() ? cur : { e: tgt.e, err: tgt.err ?? 0, pulse: 0, tint: [...tgt.tint] };

    /* ---------- the shader ----------
       The colours are the theme's primitives from lib/sky.ts, written into
       the source, so the curtains and the horizon can never disagree. */
    const FRAG = [
      "#ifdef GL_FRAGMENT_PRECISION_HIGH",
      "precision highp float;",
      "#else",
      "precision mediump float;",
      "#endif",
      "uniform vec2 r; uniform float t; uniform float e; uniform float err; uniform float pulse; uniform vec3 tint;",
      "float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }",
      "float noise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);",
      "  return mix(mix(hash(i), hash(i+vec2(1.,0.)), f.x), mix(hash(i+vec2(0.,1.)), hash(i+vec2(1.,1.)), f.x), f.y); }",
      "float fbm(vec2 p){ float v = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ v += a*noise(p); p = p*2.03 + 17.0; a *= 0.5; } return v; }",
      "void main(){",
      "  vec2 uv = gl_FragCoord.xy / r; float x = uv.x * r.x / r.y;",
      "  float warp = fbm(vec2(x*0.7 + t*0.020, t*0.035));",
      "  float band = fbm(vec2(x*1.25 + warp*1.7, t*0.045 + 3.0));",
      "  float rays = noise(vec2(x*11.0 + warp*7.0, uv.y*0.55 - t*0.16));",
      "  float fine = noise(vec2(x*31.0 + warp*9.0, uv.y*1.4 - t*0.3));",
      "  float lift = -0.05 - 0.45*e;",
      "  float height = smoothstep(lift, 1.3, uv.y + (band - 0.5)*0.7);",
      /* a soft floor under the band, so a lull in the weather thins the
         curtains without emptying the sky; it backs off as energy rises,
         where the dark gaps between curtains are the point */
      "  float fl = 0.44 - 0.14*e, dd = band - fl;",
      "  float curtain = pow(fl + 0.5*(dd + sqrt(dd*dd + 0.004)), 1.7) * (0.5 + 0.38*rays + 0.12*fine) * height;",
      `  vec3 teal = ${glslVec3(TEAL)}, violet = ${glslVec3(VIOLET)}, magenta = ${glslVec3(MAGENTA)};`,
      "  vec3 col = mix(teal, violet, smoothstep(0.30, 0.85, uv.y + (band - 0.5)*0.5));",
      "  col = mix(col, magenta, smoothstep(0.72, 1.1, uv.y + rays*0.25)*0.55);",
      "  col = mix(col, tint, 0.5);",
      `  col = mix(col, ${glslVec3(ERROR_WASH)}, err*0.8);`,
      "  float a = curtain * (0.7 + 2.3*e + 0.8*pulse);",
      `  vec3 night = ${glslVec3(NIGHT)};`,
      "  vec3 c = night + col*min(a, 0.92) + tint*0.06*e*(1.0 - uv.y);",
      "  c += (hash(gl_FragCoord.xy + t) - 0.5) / 255.0;",
      "  gl_FragColor = vec4(c, 1.0);",
      "}",
    ].join("\n");

    let gl: WebGLRenderingContext | null = null;
    const U: Record<string, WebGLUniformLocation | null> = {};
    let raf = 0;
    let last = 0;
    /* T is the weather clock. It starts at a moment measured to be lit on
       both sides of the column across common window shapes, so the first
       impression is never a lull. */
    let T = 100;

    const resizeShader = () => {
      if (!gl) return;
      const [w, h] = shaderSize(window.innerWidth, window.innerHeight);
      glCanvas.width = w;
      glCanvas.height = h;
      gl.viewport(0, 0, w, h);
    };
    const initGL = (): boolean => {
      try {
        gl = glCanvas.getContext("webgl", {
          antialias: false,
          alpha: false,
          powerPreference: "low-power",
        });
      } catch {
        gl = null;
      }
      if (!gl || gl.isContextLost()) {
        gl = null;
        return false;
      }
      const g = gl;
      const shader = (type: number, src: string) => {
        const s = g.createShader(type);
        if (!s) return null;
        g.shaderSource(s, src);
        g.compileShader(s);
        return g.getShaderParameter(s, g.COMPILE_STATUS) ? s : null;
      };
      const vs = shader(g.VERTEX_SHADER, "attribute vec2 p; void main(){ gl_Position = vec4(p, 0., 1.); }");
      const fs = shader(g.FRAGMENT_SHADER, FRAG);
      const prog = g.createProgram();
      if (!vs || !fs || !prog) {
        gl = null;
        return false;
      }
      g.attachShader(prog, vs);
      g.attachShader(prog, fs);
      g.linkProgram(prog);
      if (!g.getProgramParameter(prog, g.LINK_STATUS)) {
        gl = null;
        return false;
      }
      g.useProgram(prog);
      g.bindBuffer(g.ARRAY_BUFFER, g.createBuffer());
      g.bufferData(g.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), g.STATIC_DRAW);
      const loc = g.getAttribLocation(prog, "p");
      g.enableVertexAttribArray(loc);
      g.vertexAttribPointer(loc, 2, g.FLOAT, false, 0, 0);
      for (const n of ["r", "t", "e", "err", "pulse", "tint"]) U[n] = g.getUniformLocation(prog, n);
      resizeShader();
      return true;
    };
    const draw = () => {
      if (!gl) return;
      gl.uniform2f(U.r, glCanvas.width, glCanvas.height);
      gl.uniform1f(U.t, T);
      gl.uniform1f(U.e, cur.e);
      gl.uniform1f(U.err, cur.err);
      gl.uniform1f(U.pulse, cur.pulse);
      gl.uniform3f(U.tint, cur.tint[0], cur.tint[1], cur.tint[2]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    const shaderRunning = () => gl !== null && raf !== 0;
    /* The GPU can take the context away (a reset, a tab put to sleep on a
       crowded device). The curtains take over at once; when the context comes
       back the program is rebuilt and the shader resumes. */
    const onContextLost = (e: Event) => {
      e.preventDefault();
      gl = null;
      start();
    };
    const onContextRestored = () => start();
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      let dt = now - last;
      if (!frameDue(dt, shaderFrameInterval(settled(), tgt.e))) return;
      last = now;
      dt = Math.min(dt, 120);
      stepWeather(dt);
      T += (dt / 1000) * (0.75 + 2.4 * cur.e);
      draw();
    };
    const stopShader = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    };

    /* ---------- the CSS curtains ---------- */
    const applyCss = () => {
      host.style.setProperty("--sky-e", String(tgt.e));
      host.querySelectorAll<HTMLElement>(".sky-css i").forEach((layer) => {
        layer.getAnimations().forEach((a) => a.updatePlaybackRate(0.6 + 5 * tgt.e));
      });
    };

    /* ---------- the renderer in use ---------- */
    let mode: SkyRenderer = "off";
    const conn = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection;
    const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    const pick = () =>
      pickSkyRenderer({
        reducedTransparency: reduceTransparency.matches,
        saveData: conn?.saveData === true,
        deviceMemory: typeof memory === "number" ? memory : null,
        webgl: gl !== null || initGL(),
      });
    const start = () => {
      stopShader();
      stars.stop();
      mode = pick();
      root.dataset.sky = mode;
      if (mode === "shader") {
        if (reduceMotion.matches) {
          snap();
          draw();
        } else {
          last = performance.now();
          /* the first curtain frame now, in the same task as the stars'
             first: the program's first draw carries its compile, and until
             it was done a frame of stars on black stood for ~200ms */
          draw();
          raf = requestAnimationFrame(loop);
        }
      } else if (mode === "css") {
        applyCss();
      }
      stars.sync();
    };
    const setState = (s: string) => {
      const d = weatherFor(s);
      tgt = d;
      if (d.pulse) cur.pulse = 1;
      const tint = cssRgb(d.tint);
      root.style.setProperty("--sky-tint", tint);
      host.style.setProperty("--sky-tint", tint);
      if (mode === "css") applyCss();
      if (mode === "shader" && gl && reduceMotion.matches) {
        snap();
        draw();
        stars.still();
      }
    };

    /* ---------- the constellations ---------- */
    /* A 2D canvas, ~30 fps, above the shader. Stars live a while and are
       replaced elsewhere - sides and corners first, the centre nearly clear.
       Every so often a spark leaves one star and traces a line through a few
       neighbours, which brighten; the line rests, then the spark's tail
       catches up and the line leaves through its last star. Now and then a
       shooting star crosses the upper half. Colour, brightness and pace
       follow the sky: lines take its tint, a busy sky (thinking) traces more
       often, a sky waiting on the person (confirm) rests, an error dims, and
       a finished turn's pulse lights every star for a moment. The field
       drifts a little with the thread (parallax), and the jump to the end
       streaks it outward for a moment. */
    interface Star {
      x: number; y: number; r: number; ph: number; sp: number; lit: number;
      born: number; life: number; hold: number; z: number; tint: string;
    }
    interface Line {
      path: Star[]; born: number; draw: number; hold: number; fade: number; alpha: number; tint: string;
      /** when a fast scroll carried one of its stars across the band's wrap: it fades from here */
      cut?: number;
    }
    interface Comet {
      x: number; y: number; dx: number; dy: number; born: number; dur: number; dist: number; len: number; tint: string;
    }
    interface Walk { pts: Star[]; segs: number[]; total: number }
    const stars = (() => {
      const c = starCanvas;
      const ctx = c.getContext("2d");
      const WHITE = STAR_WHITE;
      const TALL = FIELD_TALL;
      let W = 0, H = 0, sraf = 0, slast = 0, t0 = 0, nextAt = 0, cometAt = 0;
      let field: Star[] = [], lines: Line[] = [], comets: Comet[] = [];
      /* the field spans the visible pane (not the window: the sidebar would
         hide a side), is a little taller than it and wraps so parallax never
         shows an edge, and is seeded half and half so neither side runs thin */
      let px0 = 0, pw = 0, oy = 0, oyT = 0, vy = 0, vs = 0, warpAt = -1e9, paneAt = -1e9;
      const X = (st: Star) => px0 + st.x * pw;
      const Y = (st: Star) => fieldY(st.y * H, oy * st.z, H, TALL);
      const thinSide = () => {
        let l = 0;
        for (const st of field) if (st.x < 0.5) l++;
        return l * 2 > field.length ? 1 : -1;
      };
      /* a place for a star on the given side, weighted to the sides and a
         touch more to the corners. The weight is the horizontal distance
         from the centre, so the centre column never lights: the band wraps
         vertically, so a row seeded "at the top edge" is in time drawn
         anywhere, behind the prose included */
      const place = (st: Star, now: number, side: number): Star => {
        let x: number, y: number, edge: number;
        do {
          x = (side < 0 ? 0 : 0.5) + Math.random() * 0.5;
          y = Math.random() * TALL;
          edge = Math.abs(x - 0.5) * 2 * (0.75 + 0.5 * Math.abs(y / TALL - 0.5));
        } while (Math.random() > Math.pow(edge, 2));
        st.x = x; st.y = y; st.r = 0.6 + Math.random() * 1.5; st.ph = Math.random() * 6.283;
        st.sp = 0.5 + Math.random() * 1.3; st.lit = 0; st.born = now; st.life = 9000 + Math.random() * 26000;
        st.hold = 0; st.z = 0.45 + Math.random() * 0.9; st.tint = Math.random() < 0.3 ? "sky" : WHITE;
        return st;
      };
      const blank = (): Star => ({ x: 0, y: 0, r: 1, ph: 0, sp: 1, lit: 0, born: 0, life: 1, hold: 0, z: 1, tint: WHITE });
      const seed = (now: number) => {
        field = []; lines = []; comets = [];
        const n = Math.round((W * H * TALL) / 8500);
        for (let i = 0; i < n; i++) {
          const st = place(blank(), now, i % 2 ? 1 : -1);
          st.born = now - Math.random() * st.life;
          field.push(st);
        }
      };
      const ends = (st: Star) => Math.max(st.life, st.hold - st.born);
      /* a star fades in, lives (longer if a line is using it), fades out */
      const envelope = (st: Star, now: number) => {
        const age = now - st.born, end = ends(st);
        if (age < 1500) return age / 1500;
        if (age > end) return Math.max(0, 1 - (age - end) / 2500);
        return 1;
      };
      /* the visible pane: the chat or console column, or the whole window */
      const pane = (now: number) => {
        if (now - paneAt < 250) return;
        paneAt = now;
        const el = document.querySelector<HTMLElement>(".main-col, .console-main");
        const r = el?.getBoundingClientRect();
        px0 = r ? Math.max(0, r.left) : 0;
        pw = W - px0;
      };
      const size = () => {
        if (!ctx) return;
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        W = c.clientWidth; H = c.clientHeight;
        c.width = W * dpr; c.height = H * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        paneAt = -1e9;
        pane(performance.now());
        seed(performance.now());
      };
      const spawn = (now: number, sky: Weather) => {
        const settledStars = field.filter((st) => {
          const age = now - st.born;
          return age > 1500 && age < ends(st) - 600;
        });
        if (settledStars.length < 4) return;
        /* a few tries at a starting star: one in a sparse spot has no neighbours in reach */
        let path: Star[] = [];
        const want = 3 + Math.floor(Math.random() * 4);
        let tries = 8;
        do {
          path = [settledStars[Math.floor(Math.random() * settledStars.length)]];
          while (path.length < want) {
            const at = path[path.length - 1];
            let best: Star | null = null, bd = 1e9;
            for (const st of settledStars) {
              if (path.includes(st)) continue;
              /* as drawn, not as placed: across the band's wrap two stars
                 are neighbours in the field and a window apart on screen */
              const dx = X(st) - X(at), dy = Y(st) - Y(at), d = dx * dx + dy * dy;
              if (d < bd) { bd = d; best = st; }
            }
            if (!best || Math.sqrt(bd) > Math.min(W, H) * 0.24) break;
            path.push(best);
          }
        } while (path.length < 3 && --tries);
        if (path.length < 3) return;
        const l: Line = {
          path, born: now, draw: 900 + Math.random() * 800, hold: 1600 + Math.random() * 2600,
          fade: 1100 + Math.random() * 700,
          alpha: Math.random() < 0.35 ? 0.55 + Math.random() * 0.3 : 0.2 + Math.random() * 0.16,
          tint: Math.random() < 0.7 ? tripletRgb(sky.tint) : WHITE,
        };
        for (const st of path) st.hold = Math.max(st.hold, now + l.draw + l.hold + l.fade + 800);
        lines.push(l);
      };
      const glowAt = (x: number, y: number, tint: string, a: number, r: number) => {
        if (!ctx || a <= 0) return;
        const g = ctx.createRadialGradient(x, y, 0, x, y, r * 4);
        g.addColorStop(0, `rgba(${tint},${a})`);
        g.addColorStop(0.3, `rgba(${tint},${a * 0.35})`);
        g.addColorStop(1, `rgba(${tint},0)`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(x, y, r * 4, 0, 6.283); ctx.fill();
      };
      /* the part of a line between two fractions of its length, and the point at one */
      const walk = (l: Line): Walk => {
        const pts = l.path, segs: number[] = [];
        let total = 0;
        for (let i = 1; i < pts.length; i++) {
          const d = Math.hypot(X(pts[i]) - X(pts[i - 1]), Y(pts[i]) - Y(pts[i - 1]));
          segs.push(d); total += d;
        }
        return { pts, segs, total };
      };
      const pointAt = (w: Walk, f: number) => {
        let left = f * w.total;
        for (let i = 0; i < w.segs.length; i++) {
          if (left <= w.segs[i] || i === w.segs.length - 1) {
            const k = Math.min(1, left / w.segs[i]), a = w.pts[i], b = w.pts[i + 1];
            return { x: X(a) + (X(b) - X(a)) * k, y: Y(a) + (Y(b) - Y(a)) * k };
          }
          left -= w.segs[i];
        }
        return { x: X(w.pts[0]), y: Y(w.pts[0]) };
      };
      const strokeLine = (l: Line, w: Walk, from: number, to: number, fade: number) => {
        if (!ctx) return;
        const a = from * w.total, b = to * w.total;
        let acc = 0, started = false;
        const seam = Math.min(W, H) * 0.5;
        ctx.beginPath();
        for (let i = 0; i < w.segs.length; i++) {
          const s0 = acc, s1 = acc + w.segs[i];
          acc = s1;
          if (s1 < a || s0 > b) continue;
          /* a leg across the band's wrap would cross the window: left out */
          if (w.segs[i] > seam) { started = false; continue; }
          const p = w.pts[i], q = w.pts[i + 1], ax = X(p), ay = Y(p), bx = X(q), by = Y(q);
          const ka = Math.max(0, (a - s0) / w.segs[i]), kb = Math.min(1, (b - s0) / w.segs[i]);
          if (!started) { ctx.moveTo(ax + (bx - ax) * ka, ay + (by - ay) * ka); started = true; }
          ctx.lineTo(ax + (bx - ax) * kb, ay + (by - ay) * kb);
          if (ka <= 0) p.lit = Math.max(p.lit, fade * l.alpha);
          if (kb >= 1) q.lit = Math.max(q.lit, fade * l.alpha);
        }
        ctx.lineWidth = 2.6; ctx.strokeStyle = `rgba(${l.tint},${l.alpha * fade * 0.16})`; ctx.stroke();
        ctx.lineWidth = 1; ctx.strokeStyle = `rgba(${l.tint},${l.alpha * fade})`; ctx.stroke();
      };
      const ease = (p: number) => 1 - Math.pow(1 - p, 3);
      const drawLine = (l: Line, now: number): boolean => {
        const age = now - l.born, w = walk(l), glow = Math.min(1, l.alpha + 0.4);
        if (age > l.draw + l.hold + l.fade) return false;
        /* a fast scroll can carry one of its stars across the band's wrap;
           the leg across is left out and the rest of the line fades away */
        if (l.cut === undefined && w.segs.some((s) => s > Math.min(W, H) * 0.5)) l.cut = now;
        const cut = l.cut === undefined ? 1 : Math.max(0, 1 - (now - l.cut) / 350);
        if (cut <= 0) return false;
        if (age < l.draw) {
          /* the spark traces the line in, and dims as it arrives */
          const p = age / l.draw, to = ease(p);
          strokeLine(l, w, 0, to, cut);
          const head = pointAt(w, to);
          glowAt(head.x, head.y, l.tint, glow * cut * Math.min(1, p * 4) * Math.min(1, (1 - p) * 3), 1.8);
        } else if (age < l.draw + l.hold) {
          strokeLine(l, w, 0, 1, cut);
        } else {
          /* the tail catches up; the line leaves through its last star, which flares */
          const f = (age - l.draw - l.hold) / l.fade, from = ease(f), fade = (1 - from * 0.5) * cut;
          strokeLine(l, w, from, 1, fade);
          const head = pointAt(w, from);
          glowAt(head.x, head.y, l.tint, (glow - 0.1) * cut * Math.min(1, f * 4) * (1 - from), 1.4);
          const lastStar = l.path[l.path.length - 1];
          lastStar.lit = Math.max(lastStar.lit, from * from * 0.9);
        }
        return true;
      };
      /* shooting stars: a streak from the upper part of the sky, a clear diagonal, gone in about a second */
      const comet = (now: number, sky: Weather) => {
        let x: number;
        const ang = Math.PI / 2 + (Math.random() < 0.5 ? -1 : 1) * (0.5 + Math.random() * 0.55);
        do { x = Math.random(); } while (Math.random() > Math.pow(Math.abs(x - 0.5) * 2, 1.5));
        comets.push({
          x: x * W, y: (0.03 + Math.random() * 0.58) * H, dx: Math.cos(ang), dy: Math.sin(ang), born: now,
          dur: 950 + Math.random() * 600, dist: Math.min(W, H) * (0.2 + Math.random() * 0.16),
          len: 90 + Math.random() * 70, tint: Math.random() < 0.5 ? tripletRgb(sky.tint) : WHITE,
        });
      };
      const drawComet = (k: Comet, now: number): boolean => {
        if (!ctx) return false;
        const p = (now - k.born) / k.dur;
        if (p > 1) return false;
        const hx = k.x + k.dx * k.dist * p, hy = k.y + oy + k.dy * k.dist * p;
        const a = Math.pow(Math.sin(Math.PI * p), 1.4) * 0.9, tl = k.len * Math.min(1, p / 0.3, (1 - p) / 0.35);
        const g = ctx.createLinearGradient(hx, hy, hx - k.dx * tl, hy - k.dy * tl);
        g.addColorStop(0, `rgba(${k.tint},${a})`); g.addColorStop(1, `rgba(${k.tint},0)`);
        ctx.strokeStyle = g; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx - k.dx * tl, hy - k.dy * tl); ctx.stroke();
        glowAt(hx, hy, k.tint, a, 1.3);
        return true;
      };
      /* ---------- the chart of rings ----------
         Behind the boot's words and, quieter, behind the new chat's greeting:
         an instrument ring turning slowly against the clock with sparkles
         riding it, an outer ring broken into arcs turning with it, dust
         beyond, and three orbits each with a head and a tail. A feathered
         cut behind the words keeps the rings off the text. */
      const GREY = WHITE;
      const VIOLET_T = tripletRgb(VIOLET), AMBER_T = tripletRgb(AMBER);
      interface Orbit { r: number; tilt: number; rot: number; speed: number; span: number; color: string }
      const ORBITS: Orbit[] = [
        { r: 1.1, tilt: 0.42, rot: -0.35, speed: 0.36, span: 2.6, color: "sky" },
        { r: 1.3, tilt: 0.3, rot: 0.55, speed: -0.22, span: 2.3, color: VIOLET_T },
        { r: 0.96, tilt: 0.62, rot: 1.3, speed: 0.5, span: 1.9, color: AMBER_T },
      ];
      const glowOn = (cx: CanvasRenderingContext2D, x: number, y: number, tint: string, a: number, r: number) => {
        if (a <= 0) return;
        const g = cx.createRadialGradient(x, y, 0, x, y, r * 4);
        g.addColorStop(0, `rgba(${tint},${a})`);
        g.addColorStop(0.3, `rgba(${tint},${a * 0.35})`);
        g.addColorStop(1, `rgba(${tint},0)`);
        cx.fillStyle = g;
        cx.beginPath(); cx.arc(x, y, r * 4, 0, 6.283); cx.fill();
      };
      /* a four-point star, as charts draw them */
      const sparkle = (cx: CanvasRenderingContext2D, x: number, y: number, r: number, tint: string, a: number) => {
        glowOn(cx, x, y, tint, a * 0.5, r * 0.45);
        cx.fillStyle = `rgba(${tint},${a})`;
        cx.beginPath(); cx.moveTo(x, y - r);
        cx.quadraticCurveTo(x, y, x + r, y); cx.quadraticCurveTo(x, y, x, y + r);
        cx.quadraticCurveTo(x, y, x - r, y); cx.quadraticCurveTo(x, y, x, y - r); cx.fill();
      };
      const ring = (cx: CanvasRenderingContext2D, r: number, a: string, w: number) => {
        cx.lineWidth = w; cx.strokeStyle = a; cx.beginPath(); cx.arc(0, 0, r, 0, 6.283); cx.stroke();
      };
      const ticks = (cx: CanvasRenderingContext2D, r: number, n: number, every: number, a: string) => {
        cx.lineWidth = 1; cx.strokeStyle = a; cx.beginPath();
        for (let i = 0; i < n; i++) {
          const ang = (i * 6.283) / n, r0 = r - (i % every ? 4 : 11);
          cx.moveTo(Math.cos(ang) * r0, Math.sin(ang) * r0); cx.lineTo(Math.cos(ang) * r, Math.sin(ang) * r);
        }
        cx.stroke();
      };
      const orbit = (cx: CanvasRenderingContext2D, o: Orbit, R: number, t: number, skyTint: string) => {
        const color = o.color === "sky" ? skyTint : o.color, r = o.r * R, head = t * o.speed, N = 48;
        cx.save(); cx.rotate(o.rot); cx.scale(1, o.tilt);
        ring(cx, r, `rgba(${color},.16)`, 1);
        const back = o.speed < 0 ? -1 : 1;   /* the tail lies behind the head, whichever way it travels */
        cx.lineWidth = 9; cx.strokeStyle = `rgba(${color},.10)`; cx.beginPath();
        cx.arc(0, 0, r, head - back * o.span * 0.4, head, back < 0); cx.stroke();
        for (let i = 0; i < N; i++) {   /* the tail: alpha and width ramp up toward the head */
          const k = (i + 1) / N, a0 = head - back * o.span * (1 - i / N), a1 = head - back * o.span * (1 - k);
          cx.lineWidth = 0.6 + 2.4 * k; cx.strokeStyle = `rgba(${color},${k * k * 0.95})`;
          cx.beginPath(); cx.arc(0, 0, r, a0, a1 + back * 0.015, back < 0); cx.stroke();
        }
        cx.restore();
        const ex = Math.cos(head) * r, ey = Math.sin(head) * r * o.tilt, cs = Math.cos(o.rot), sn = Math.sin(o.rot);
        const hx = ex * cs - ey * sn, hy = ex * sn + ey * cs;
        sparkle(cx, hx, hy, 8, color, 0.95); glowOn(cx, hx, hy, color, 0.8, 2.2);
      };
      const chart = (cx: CanvasRenderingContext2D, t: number, sky: Weather, mx: number, my: number, R: number, dim: number) => {
        const tint = tripletRgb(sky.tint), breathe = 1 + 0.012 * Math.sin(t * 0.9);
        cx.save(); cx.globalAlpha = dim; cx.translate(mx, my); cx.scale(breathe, breathe);
        /* the instrument ring, turning slowly against the clock, with sparkles riding it */
        cx.save(); cx.rotate(-t * 0.01);
        ring(cx, R, `rgba(${GREY},.26)`, 1); ring(cx, R, `rgba(${tint},.08)`, 6); ring(cx, R - 14, `rgba(${GREY},.12)`, 1);
        ticks(cx, R, 120, 10, `rgba(${GREY},.34)`);
        for (let i = 0; i < 6; i++) {
          const ang = (i * 6.283) / 6 + 0.4, tw = 0.45 + 0.55 * Math.abs(Math.sin(t * 1.6 + i * 1.3));
          sparkle(cx, Math.cos(ang) * R, Math.sin(ang) * R, 5 + 2 * tw, i % 2 ? GREY : tint, tw);
        }
        cx.restore();
        /* an outer ring broken into eight arcs, turning with the clock, dust beyond it */
        cx.save(); cx.rotate(t * 0.016);
        for (let k = 0; k < 8; k++) {
          cx.lineWidth = 1; cx.strokeStyle = `rgba(${GREY},.2)`; cx.beginPath();
          cx.arc(0, 0, R * 1.2, (k * Math.PI) / 4, (k * Math.PI) / 4 + (Math.PI / 4) * 0.72); cx.stroke();
        }
        cx.fillStyle = `rgba(${GREY},.3)`;
        for (let d = 0; d < 48; d++) {
          const da = (d * 6.283) / 48;
          cx.beginPath(); cx.arc(Math.cos(da) * R * 1.24, Math.sin(da) * R * 1.24, d % 4 ? 0.8 : 1.4, 0, 6.283); cx.fill();
        }
        cx.restore();
        for (const o of ORBITS) orbit(cx, o, R, t, tint);
        cx.restore();
        /* a feathered cut behind the words: the rings pass there dimly, not over the text */
        cx.save(); cx.globalCompositeOperation = "destination-out"; cx.translate(mx, my); cx.scale(1, 0.82);
        const hole = cx.createRadialGradient(0, 0, 0, 0, 0, R * 0.92);
        hole.addColorStop(0, "rgba(0,0,0,.8)"); hole.addColorStop(0.62, "rgba(0,0,0,.65)"); hole.addColorStop(1, "rgba(0,0,0,0)");
        cx.fillStyle = hole; cx.beginPath(); cx.arc(0, 0, R * 0.92, 0, 6.283); cx.fill(); cx.restore();
      };
      /* where the chart lives: the window's centre on boot, between the
         greeting and the composer on a new chat, dimmer */
      let handoffAt = -1e9;
      /* the chart arrives, never cuts: a view change re-measures the pane
         (the shell is hidden under the boot, so the pane read then is the
         whole window) and, with no handoff in flight, fades the chart in
         where it lives; a later move of its anchors (the sidebar closing,
         a resize) is eased, not stepped. The still frame snaps: its clock
         never advances, so an ease there would never arrive. */
      let lastView = readView(), arriveAt = -1e9;
      /* and it leaves as it arrives: into a thread, out over 260ms where it
         stood (shown is how bright that was), beside the greeting's fade */
      let departAt = -1e9, shown = 0;
      let gx = 0, gy = 0, gR = 0, gAt = -1e9;
      const follow = (now: number, x: number, y: number, R: number, snap: boolean) => {
        const dt = Math.max(0, now - gAt);
        const k = snap || reduceMotion.matches || dt > 1000 ? 1 : 1 - Math.exp(-dt / 160);
        gx += (x - gx) * k; gy += (y - gy) * k; gR += (R - gR) * k; gAt = now;
      };
      const bootGeometry = () => ({ x: W / 2, y: H * 0.48, R: Math.min(W * 0.34, H * 0.46), dim: 1 });
      const newGeometry = () => {
        const a = document.querySelector(".welcome")?.getBoundingClientRect();
        const b = document.querySelector(".composer-box")?.getBoundingClientRect();
        const y = a && b ? (a.top + b.bottom) / 2 : H * 0.44;
        return { x: px0 + pw / 2, y, R: Math.min(pw * 0.31, H * 0.44), dim: 0.55 };
      };
      /* the boot's own canvas, above its pool of shade, sized to the window */
      let bootCanvas: HTMLCanvasElement | null = null, bctx: CanvasRenderingContext2D | null = null;
      const bootContext = (): CanvasRenderingContext2D | null => {
        const el = document.querySelector<HTMLCanvasElement>(".boot-chart");
        if (!el) { bootCanvas = null; bctx = null; return null; }
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        if (el !== bootCanvas || el.width !== Math.round(W * dpr) || el.height !== Math.round(H * dpr)) {
          bootCanvas = el; el.width = Math.round(W * dpr); el.height = Math.round(H * dpr);
          bctx = el.getContext("2d");
          bctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        return bctx;
      };
      const drawChart = (now: number, t: number, sky: Weather) => {
        if (!ctx) return;
        const view = readView();
        const moved = view !== lastView;
        if (moved) {
          if (lastView === "new" && view === "thread" && !reduceMotion.matches) departAt = now;
          lastView = view;
          paneAt = -1e9;
          pane(now);
          if ((view === "new" || view === "login") && !reduceMotion.matches && now - handoffAt > 1200) arriveAt = now;
        }
        if (view === "boot") {
          const cx = bootContext();
          if (!cx) return;
          cx.clearRect(0, 0, W, H);
          const g = bootGeometry();
          chart(cx, t, sky, g.x, g.y, g.R, 1);
          return;
        }
        /* the sign-in: the chart at the boot's place and its full brightness,
           behind the card, in over 560ms as it arrives; gone under 560px, as
           the boot's is (a card over a chart that wide reads as clutter) */
        if (view === "login") {
          if (W < 560) return;
          const g = bootGeometry();
          const da = now - arriveAt;
          let dim = 1;
          if (da < 560) { const f = da / 560; dim = f * f * (3 - 2 * f); }
          if (dim > 0.01) chart(ctx, t, sky, g.x, g.y, g.R, dim);
          return;
        }
        if (view !== "new") {
          const dd = now - departAt;
          if (view === "thread" && dd >= 0 && dd < 260) {
            let f = 1 - dd / 260; f = f * f * (3 - 2 * f);
            if (shown * f > 0.01) chart(ctx, t, sky, gx, gy, gR, shown * f);
          }
          return;
        }
        /* arriving from boot: the chart fades where it was, rests a beat, then fades in where it lives now */
        let g = newGeometry(), dim = g.dim;
        const dt = now - handoffAt;
        /* snapped through the handoff too, so the eased place is the live one
           the moment the choreography hands over */
        follow(now, g.x, g.y, g.R, moved || dt < 980);
        if (dt < 980) {
          let f: number;
          if (dt < 260) { f = 1 - dt / 260; f = f * f * (3 - 2 * f); g = bootGeometry(); dim = f; }
          else if (dt < 420) dim = 0;
          else { f = (dt - 420) / 560; f = f * f * (3 - 2 * f); dim = g.dim * f; }
          if (dim > 0.01) chart(ctx, t, sky, g.x, g.y, g.R, dim);
          shown = dt < 260 ? 0 : dim;
          return;
        }
        /* a quick load, or a new chat opened from a thread: in over 560ms */
        const da = now - arriveAt;
        if (da < 560) { let f = da / 560; f = f * f * (3 - 2 * f); dim *= f; }
        if (dim > 0.01) chart(ctx, t, sky, gx, gy, gR, dim);
        shown = dim;
      };
      const handoff = () => { handoffAt = performance.now(); };
      const drawStars = (now: number, t: number, sky: Weather) => {
        if (!ctx) return;
        const bright = (0.7 + 0.5 * sky.e) * (1 - 0.35 * sky.err);
        /* the jump: the field is already rushing up with the glide; for a
           moment each star also leaves a soft trail below it, so the view
           reads as falling */
        let wv = Math.sin(Math.PI * Math.min(1, (now - warpAt) / 1100));
        if (wv < 0) wv = 0;
        vs += (Math.abs(vy) - vs) * (Math.abs(vy) > vs ? 0.22 : 0.1);
        const trail = vs * (2.2 + 6 * wv), dir = vy < 0 ? 1 : -1;
        let ts = Math.min(1, Math.max(0, (trail - 0.6) / 3));
        ts = ts * ts * (3 - 2 * ts);
        for (const st of field) {
          let env = envelope(st, now);
          if (env <= 0 && now - st.born > 2000) { place(st, now, thinSide()); env = 0; }
          const tw = 0.5 + 0.5 * Math.sin(t * st.sp + st.ph);
          const a = ((0.28 + 0.55 * tw) * bright + sky.pulse * 0.4 + wv * 0.3 * ts) * env;
          const tint = st.tint === "sky" ? tripletRgb(sky.tint) : st.tint, sx = X(st), sy = Y(st);
          /* the larger stars carry a soft halo, so the field reads as depth and not as grain */
          if (st.r > 1.6) glowAt(sx, sy, tint, Math.min(1, a + st.lit) * 0.3, st.r * 0.9);
          if (ts > 0) {
            const tl = Math.min(34, trail * st.z * st.z * 1.6), ta = Math.min(1, a + st.lit) * ts, ey = sy + dir * tl;
            const g = ctx.createLinearGradient(sx, sy, sx, ey);
            g.addColorStop(0, `rgba(${tint},${ta * 0.7})`); g.addColorStop(1, `rgba(${tint},0)`);
            ctx.strokeStyle = g; ctx.lineCap = "round";
            ctx.lineWidth = st.r * 5; ctx.globalAlpha = 0.28;
            ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, ey); ctx.stroke();
            ctx.lineWidth = st.r * 1.5; ctx.globalAlpha = 1;
            ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, ey); ctx.stroke();
          }
          ctx.fillStyle = `rgba(${tint},${Math.min(1, a + st.lit)})`;
          ctx.beginPath(); ctx.arc(sx, sy, st.r + st.lit * 1.4, 0, 6.283); ctx.fill();
          st.lit *= 0.9;
        }
      };
      /* ms until the next constellation: a busy sky traces more, a sky waiting on the person rests */
      const pace = (sky: Weather) => {
        const base = readTurnState() === "confirm" ? 4200 : 2100 - 1400 * sky.e;
        return base + Math.random() * base;
      };
      const frame = (now: number) => {
        sraf = requestAnimationFrame(frame);
        /* 30 fps at rest; full rate while something moves fast enough to step: a jump, a quick scroll */
        const fast = now - warpAt < 1200 || now - handoffAt < 1200 || now - departAt < 300 || Math.abs(vy) > 0.4;
        if (!fast && !frameDue(now - slast, STARS_FRAME_INTERVAL)) return;
        slast = now;
        if (!ctx) return;
        const sky = current(), t = (now - t0) / 1000;
        pane(now);
        /* parallax: the field follows the thread at a tenth, eased, so
           streamed lines never jog it; a fast glide moves it fast and the
           trails show. Unbounded: the band wraps, so a long conversation's
           every scroll still moves the sky */
        const thread = document.querySelector<HTMLElement>('[data-part="thread"]');
        oyT = thread ? parallaxOffset(thread.scrollTop) : 0;
        const was = oy;
        oy += (oyT - oy) * 0.16;
        vy = oy - was;
        ctx.clearRect(0, 0, W, H);
        /* the chart is drawn first, so its cut behind the words spares the stars */
        drawChart(now, t, sky);
        const boot = readView() === "boot";
        if (now > nextAt) { spawn(now, sky); nextAt = now + pace(sky) * (boot ? 0.6 : 1); }
        if (now > cometAt) {
          comet(now, sky);
          cometAt = now + (boot ? 2500 + Math.random() * 3500 : 5000 + Math.random() * 8000 - 4000 * sky.e);
        }
        lines = lines.filter((l) => drawLine(l, now));
        comets = comets.filter((k) => drawComet(k, now));
        drawStars(now, t, sky);
      };
      /* still stars, nothing moving: the reduced-motion rendering */
      const still = () => {
        if (!ctx || !W) return;
        ctx.clearRect(0, 0, W, H);
        /* a still chart too, where the view has one; the handoff is a cut */
        handoffAt = -1e9;
        drawChart(t0 + 2000, 0, current());
        drawStars(t0 + 2000, 0, current());
      };
      const start = () => {
        if (sraf || !c.offsetParent) return;
        size();
        /* a sky switched back on has no chart to fade: what it showed before
           it stopped is long gone */
        shown = 0;
        t0 = performance.now(); nextAt = t0 + 400; cometAt = t0 + 1500;
        if (reduceMotion.matches) { still(); return; }
        sraf = requestAnimationFrame(frame);
      };
      const stop = () => {
        if (sraf) cancelAnimationFrame(sraf);
        sraf = 0;
      };
      const sync = () => {
        if (getComputedStyle(c).display === "none") stop();
        else start();
      };
      const warp = () => {
        if (!reduceMotion.matches && sraf) warpAt = performance.now();
      };
      const resize = () => {
        if (sraf) size();
        else if (reduceMotion.matches && c.offsetParent) { size(); still(); }
      };
      return { sync, stop, warp, resize, still, handoff };
    })();

    /* ---------- wiring ---------- */
    setState(readTurnState());
    start();
    const unobserve = observeTurnState(setState);
    /* under reduced motion nothing loops, so a view change (the boot's
       handoff, a new chat) redraws the still frame with its chart */
    const unobserveView = observeView(() => {
      if (reduceMotion.matches) stars.still();
    });
    const onResize = () => {
      resizeShader();
      /* setting the canvas size clears it; paint now rather than at the next
         capped frame, so dragging the window edge never shows the night */
      if (gl) draw();
      stars.resize();
    };
    const onVisibility = () => {
      if (document.hidden) { stopShader(); stars.stop(); }
      else start();
    };
    const onJump = () => stars.warp();
    const onHandoff = () => stars.handoff();
    window.addEventListener("resize", onResize);
    document.addEventListener("visibilitychange", onVisibility);
    document.addEventListener(SKY_JUMP_EVENT, onJump);
    document.addEventListener(SKY_HANDOFF_EVENT, onHandoff);
    glCanvas.addEventListener("webglcontextlost", onContextLost);
    glCanvas.addEventListener("webglcontextrestored", onContextRestored);
    reduceMotion.addEventListener("change", start);
    reduceTransparency.addEventListener("change", start);
    return () => {
      stopShader();
      stars.stop();
      unobserve();
      unobserveView();
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVisibility);
      document.removeEventListener(SKY_JUMP_EVENT, onJump);
      document.removeEventListener(SKY_HANDOFF_EVENT, onHandoff);
      reduceMotion.removeEventListener("change", start);
      reduceTransparency.removeEventListener("change", start);
      glCanvas.removeEventListener("webglcontextlost", onContextLost);
      glCanvas.removeEventListener("webglcontextrestored", onContextRestored);
      /* The context is left on the canvas, not released: a remount (the
         development double-mount, a hot reload) gets the same canvas back and
         getContext hands it the same context, whereas a released one comes
         back lost and the shader would silently fall to the curtains. The
         context goes with the canvas when the element does. */
      root.style.removeProperty("--sky-tint");
    };
  }, []);

  return (
    <div ref={rootRef} className="sky" aria-hidden="true">
      <canvas ref={glRef} className="sky-gl" />
      <canvas ref={starsRef} className="sky-stars" />
      <div className="sky-css">
        <i />
        <i />
        <i />
      </div>
    </div>
  );
}
