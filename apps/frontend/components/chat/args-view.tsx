"use client";

/** Render tool-call arguments as readable key/value lines (strings as-is, else JSON). */
export function ArgsView({ args }: { args: Record<string, unknown> }) {
  const entries = Object.entries(args);
  if (entries.length === 0) return null;
  return (
    <div className="args-view">
      {entries.map(([key, value]) => (
        <div key={key} className="arg-row">
          <span className="arg-key">{key}</span>
          <span className="arg-val">
            {typeof value === "string" ? value : JSON.stringify(value)}
          </span>
        </div>
      ))}
    </div>
  );
}
