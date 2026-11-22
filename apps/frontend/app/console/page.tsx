import { redirect } from "next/navigation";

/**
 * The console has no landing page of its own.
 *
 * A bare `/console` sends the reader to the first rail entry rather than
 * showing an empty shell that only asks them to pick again.
 */
export default function ConsolePage() {
  redirect("/console/models");
}
