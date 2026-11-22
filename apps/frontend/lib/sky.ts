/**
 * The sky's pure half: what each UI state asks of it, which renderer a
 * browser gets, and the pacing rules the loops follow.
 *
 * Everything here is data and arithmetic, so a scripted battery can hold it
 * to the dossier (docs/design/DESIGN.md, "The sky: energy and tint per state")
 * without a browser. The renderer itself lives in components/sky.tsx.
 */

/** A colour as unit RGB, the form the shader takes it in. */
export type Rgb = readonly [number, number, number];

/** What the sky shows for one UI state. */
export interface SkyWeather {
  /** How much sky: 0 is the flat night, 1 is the whole window lit. */
  e: number;
  /** The running step's colour; the curtains lean toward it. */
  tint: Rgb;
  /** How far the curtains wash toward the danger hue (0..1). */
  err?: number;
  /** Entering the state flashes every star once (done, boot ready). */
  pulse?: boolean;
}

/* The theme's primitives as the shader takes them: --p-aurora-teal,
   --p-aurora-violet, --p-aurora-magenta, --s-warn and --s-danger from
   themes/aurora.css, as unit RGB. Shader uniforms cannot read custom
   properties, so these mirror the tokens; an edit to one is an edit to both. */
/** `--p-aurora-teal`: the resting tint, and a tool's. */
export const TEAL: Rgb = [0.271, 0.878, 0.784];
/** `--p-aurora-violet`: thinking. */
export const VIOLET: Rgb = [0.604, 0.541, 0.984];
/** `--p-aurora-magenta`: a confirmation waiting on the person. */
export const MAGENTA: Rgb = [0.918, 0.498, 0.863];
/** `--s-warn`: a model loading, a download. */
export const AMBER: Rgb = [0.937, 0.725, 0.392];
/** `--s-danger`: the error tint. */
export const DANGER: Rgb = [0.941, 0.443, 0.541];
/** The deeper red the curtains themselves wash toward on an error (the tint
 * above is what the horizon and the stars take; the sky goes a shade darker). */
export const ERROR_WASH: Rgb = [0.94, 0.36, 0.46];
/** `--p-night-0`: the flat night the shader paints on. */
export const NIGHT: Rgb = [0.043, 0.051, 0.125];
/** The constellations' resting colour, as an "r,g,b" triplet. */
export const STAR_WHITE = "214,222,255";

/**
 * Energy and tint per state, as the dossier's table records them. The chat
 * page writes the turn states from its stream (loading arrives with the model
 * status); boot, sign-in and the console set the rest.
 */
export const SKY_STATES = {
  idle: { e: 0.36, tint: TEAL },
  waiting: { e: 0.48, tint: TEAL },
  loading: { e: 0.3, tint: AMBER },
  thinking: { e: 0.92, tint: VIOLET },
  tool: { e: 0.72, tint: TEAL },
  confirm: { e: 0.4, tint: MAGENTA },
  streaming: { e: 0.56, tint: TEAL },
  error: { e: 0.3, tint: DANGER, err: 1 },
  done: { e: 0.36, tint: TEAL, pulse: true },
  download: { e: 0.46, tint: AMBER },
  boot: { e: 0.5, tint: TEAL },
  "boot-model": { e: 0.84, tint: VIOLET },
  "boot-data": { e: 0.62, tint: TEAL },
  "boot-ready": { e: 0.4, tint: TEAL, pulse: true },
  "boot-slow": { e: 0.3, tint: AMBER },
  login: { e: 0.44, tint: TEAL },
} as const satisfies Record<string, SkyWeather>;

/** A state the sky knows; `<body data-state>` carries one of these. */
export type SkyState = keyof typeof SKY_STATES;

/** The weather for a state name, with anything unknown read as idle. */
export function weatherFor(state: string | null | undefined): SkyWeather {
  return state !== null && state !== undefined && state in SKY_STATES
    ? SKY_STATES[state as SkyState]
    : SKY_STATES.idle;
}

/** The three renderers; `<html data-sky>` names the one in use. */
export type SkyRenderer = "shader" | "css" | "off";

/** What the pick reads from the browser. */
export interface SkyEnvironment {
  /** `prefers-reduced-transparency: reduce` */
  reducedTransparency: boolean;
  /** The network layer asked for less (Save-Data). */
  saveData: boolean;
  /** `navigator.deviceMemory` in GB, or null where the browser withholds it. */
  deviceMemory: number | null;
  /** A WebGL context could be created and the program linked. */
  webgl: boolean;
}

/**
 * Which renderer a browser gets. The app picks, never the person: off under
 * reduced transparency (the dossier's degrade rule) or on a device that has
 * asked for less; the shader wherever WebGL works; the transform-only CSS
 * curtains as the floor beneath it.
 */
export function pickSkyRenderer(env: SkyEnvironment): SkyRenderer {
  if (env.reducedTransparency) return "off";
  if (env.saveData) return "off";
  if (env.deviceMemory !== null && env.deviceMemory <= 2) return "off";
  return env.webgl ? "shader" : "css";
}

/**
 * Milliseconds between shader frames: 30 fps while the weather is changing or
 * the turn is busy, 20 fps once it has settled at rest.
 */
export function shaderFrameInterval(settled: boolean, energy: number): number {
  return settled && energy < 0.4 ? 1000 / 20 : 1000 / 30;
}

/** The constellation canvas paces itself at 30 fps unless something is moving fast. */
export const STARS_FRAME_INTERVAL = 1000 / 30;

/**
 * How early a frame may come and still count as due. Display frames carry
 * jitter: at 60Hz two frames are 33.3ms give or take a fraction, and a cap
 * compared strictly against 33.33 lets about half of them through and
 * waits a third frame for the rest, which reads as judder on everything
 * the canvas draws.
 */
export const FRAME_TOLERANCE_MS = 2;

/** Whether a frame is due, `elapsed` ms after the last one drawn, under a cap of `interval` ms. */
export function frameDue(elapsed: number, interval: number): boolean {
  return elapsed >= interval - FRAME_TOLERANCE_MS;
}

/**
 * Move `current` toward `target` by the share a first-order ease covers in
 * `dt` milliseconds with the given time constant, so a state change reads as
 * weather settling, never as a cut.
 */
export function approach(
  current: number,
  target: number,
  dt: number,
  tau: number = 520,
): number {
  return current + (target - current) * (1 - Math.exp(-dt / tau));
}

/** A unit colour as the CSS the horizon and the curtains read. */
export function cssRgb(c: Rgb): string {
  return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
}

/** A unit colour as a GLSL vec3 literal, for building the shader source. */
export function glslVec3(c: Rgb): string {
  return `vec3(${c.map((v) => v.toFixed(3)).join(", ")})`;
}

/** A unit colour as the "r,g,b" triplet the canvas builds rgba() strings from. */
export function tripletRgb(c: Rgb): string {
  return `${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)}`;
}

/** The constellations follow the thread at this share of its scroll. */
export const PARALLAX = 0.1;

/** The star field is this much taller than the window, so parallax never shows an edge. */
export const FIELD_TALL = 1.25;

/** The field's offset for a thread scrolled `scrollTop` pixels: upward, unbounded. */
export function parallaxOffset(scrollTop: number): number {
  return -scrollTop * PARALLAX;
}

/**
 * Where a star of the field is drawn once the parallax offset is applied.
 *
 * The field is a band `tall` times the window's height, centred on it, and
 * it wraps: a star carried past one edge comes back in at the other, so the
 * stars keep answering the thread however far it has scrolled. A clamp
 * would stop them a few screens in, which on a long conversation is where
 * every scroll starts.
 *
 * @param y - The star's place in the band, 0 to `tall * height`.
 * @param offset - The parallax offset for this star (the field's, times its depth).
 * @param height - The window's height.
 * @param tall - The band's height as a share of the window's.
 * @returns The y to draw at, from `-(tall - 1) * height / 2`.
 */
export function fieldY(y: number, offset: number, height: number, tall: number = FIELD_TALL): number {
  const span = height * tall;
  if (span <= 0) return y;
  let v = (y + offset) % span;
  if (v < 0) v += span;
  return v - (height * (tall - 1)) / 2;
}

/** The shader's drawing surface: half resolution, capped at 960 wide. */
export function shaderSize(innerWidth: number, innerHeight: number): [number, number] {
  const w = Math.max(1, Math.min(960, Math.round(innerWidth * 0.5)));
  const h = Math.max(1, Math.round((w * innerHeight) / Math.max(1, innerWidth)));
  return [w, h];
}
