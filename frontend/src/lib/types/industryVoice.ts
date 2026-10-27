// One entry in the "Voices shaping modern work" ticker on Home. This is
// editorial/social-proof content about how industry leaders talk about
// hiring, interviewing, and modern work — NOT a claim that the named
// company or person uses, endorses, or partners with InterviewAI.
//
// `text` must never be presented as a verbatim quote unless it truly is
// one — see `contentType` below and the content rule in the ticker's data
// file.
export interface IndustryVoice {
  id: string;
  company: string;
  // Path to a local monochrome company logo (e.g. "/companies/nvidia.svg")
  // or, if no asset is available yet, a short glyph/initial used as a
  // stand-in badge instead (matches CompanyMarquee's approach elsewhere on
  // Home). The ticker card falls back to a text-initial badge if the path
  // fails to load, so a missing asset never renders a broken image.
  companyLogo: string;
  person: string;
  role: string;
  text: string;
  // 'quote'   — verbatim, must be traceable to `sourceUrl`.
  // 'summary' — our own paraphrase of a public statement; never rendered
  //             in quotation marks, always labeled as a summary in the UI.
  contentType: 'quote' | 'summary';
  sourceLabel: string;
  sourceUrl: string;
  sourceDate?: string;
}
