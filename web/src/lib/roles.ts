export interface RolePreset {
  id: string;
  name: string;
  summary: string;
  /** Sent as `instructions`; empty means the server's prompt alone. */
  instructions: string;
}

export const DEFAULT_ROLE = "assistant";
export const CUSTOM_ROLE = "custom";
export const MAX_ROLE_LENGTH = 1000;

export const ROLES: RolePreset[] = [
  {
    id: DEFAULT_ROLE,
    name: "Research assistant",
    summary: "Balanced, cited answers",
    instructions: "",
  },
  {
    id: "reviewer",
    name: "Skeptical reviewer",
    summary: "Weighs evidence, flags weak claims",
    instructions:
      "Act as a skeptical peer reviewer. Weigh the strength of the evidence, point out where papers disagree, " +
      "and flag claims that rest on a single study or on simulation alone.",
  },
  {
    id: "explainer",
    name: "Explain simply",
    summary: "Plain language, jargon defined",
    instructions:
      "Explain for a reader with an undergraduate physics background. Define jargon on first use, " +
      "build intuition before equations, and keep the answer short.",
  },
  {
    id: "engineer",
    name: "Reactor engineer",
    summary: "Margins, materials, plant impact",
    instructions:
      "Answer as a fusion power plant design engineer. Focus on engineering margins, materials and component " +
      "limits, and what the findings mean for building a reactor.",
  },
];
