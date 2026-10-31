# A-verification.md — verification log for A-data-protection.md

**Verified on:** 31 October 2026 (session with network access to `kenyalaw.org`/`new.kenyalaw.org`, `www.parliament.go.ke`, `libraryir.parliament.go.ke`, `www.iebc.or.ke`, `supremecourt.judiciary.go.ke`, `www.standardmedia.co.ke`, `www.the-star.co.ke`, `katibainstitute.org`, `www.kictanet.or.ke`, `khrc.or.ke`).
**Sources kept in:** `/home/user/legal-research/src/` (files named below). A small helper that queries Kenya Law's public search API is at `/home/user/legal-research/tools/klq.py`; its raw JSON results are `src/klq_*.json`.

## 0. What could and could not be reached (read this before relying on any item)

| Route | Result on 31 Oct 2026 | Consequence |
|---|---|---|
| `new.kenyalaw.org/akn/...` and `kenyalaw.org/akn/...` (document pages, `/source.pdf`) | Every request answered `HTTP 202`, zero bytes, header `x-amzn-waf-action: challenge` (AWS WAF JavaScript challenge; `src/kl_*.hdr`). I did not attempt to defeat the challenge. | Full-text pages of Acts, Regulations and judgments were **not** readable. |
| `kenyalaw.org/search/api/documents/?search=...` (Kenya Law's own public search API, not under `/akn/`) | Works. Returns, for each hit, the document title, neutral citation, date, the matched **provision heading** (e.g. "26. Requirement for specified processing to be done in Kenya") and short **verbatim snippets** (roughly 100–200 characters around each match) of the document text. | This is the same primary text as the Kenya Law page, but fragmentary. Every "quote" below marked **(KL snippet)** is verbatim from these snippets; `...` marks the API's own elisions. Where a snippet did not reach the operative words, I say so. |
| `www.odpc.go.ke` / `odpc.go.ke` | Home page returns a "Website under maintenance … temporarily unavailable" page (`src/odpc_home.html`); every sub-page and every `wp-content/uploads/...pdf` returns `404` (`src/odpc_*.pdf` are the 404 bodies). | **No ODPC document** (guidance notes, register, determinations, draft s.47 notice, press releases) could be read. Items depending on them are STILL UNVERIFIED (blocked). |
| `www.parliament.go.ke`, `libraryir.parliament.go.ke` | Reachable. Bills pages (15 pages), Bills Tracker PDF (15 Oct 2026, 80 pp.), Hansards for 15, 21 and 22 Oct 2026 downloaded (`src/parl_*`, `src/hansard_*`). DSpace REST API of the library works. | Primary evidence on bills and on the 15 Oct 2026 motion about IEBC/ODPC. |
| `www.iebc.or.ke` | Reachable. IEBC hosts Gazette copies of the Elections (Technology) Regulations 2017 and a Rev. 2017 compilation of the Elections Act subsidiary legislation (`src/iebc_*.pdf/.txt`). | Primary text for reg. 2 "biometric", reg. 79 and Form 34A. |
| `supremecourt.judiciary.go.ke` | Reachable, but its 2017 page lists only some rulings (Mwau, Thoithi, etc.), not [2017] KESC 45. `www.judiciary.go.ke` and `www.klrc.go.ke` refused at the proxy (`CONNECT 403`). | No alternative route to the Mue ruling. |
| `nation.africa`, `citizen.digital`, `posts.kictanet.or.ke`, `caselawlibrary.cipit.org` | Refused at the proxy (`CONNECT 403`) or `403`. | Nation/Citizen reports could only be seen as search-engine snippets. |
| `WebFetch` tool | Cannot resolve any of the above hosts (`ENOTFOUND`). | All fetching was done with `curl` through the session proxy. |

Labels used below: **CONFIRMED** (primary text read, report correct), **CORRECTED** (primary text read, report wrong or incomplete — corrected text given), **STILL UNVERIFIED** (reason given). "Partly confirmed" is noted inside an item where only part of the operative text was reachable.

---

## A. Cross-border transfer and data localisation

### A1. Regulation 26, Data Protection (General) Regulations 2021 (LN 263 of 2021) — CONFIRMED (two wording corrections)
- **Heading** — CORRECTED. The heading is "**Requirement for specified processing to be done in Kenya**" (the report had "Requirement for specified processing data to be done in Kenya").
- **26(1)** — CONFIRMED. KL snippets: "(1) Pursuant to section 50 of the Act, a data controller or data processor who processes personal data"; "…processor who processes personal data for the purposeof strategic interest of the state outlined under subregulation"; "…outlined under subregulation (2) shall— (a) process such personal data through a server and data centre located in Kenya"; "; or (b) store at least one serving copy of the concerned personal datain ... a data centre located in Kenya." (The Kenya Law text reads "subregulation", not "sub-regulation"; "serving copy" — not "servicing copy" as one law-firm note has it — is confirmed.)
- **26(2)(b)** — CONFIRMED verbatim: "(b) facilitating the conduct of elections for the representation of the people under the Constitution"; snippet also shows "(2) The purpose contemplated under subregulation" and "…identity management systems; (b) facilitating…". The phrase "administering of the electoral process" does not occur (query returned no hit).
- **26(3)** — CONFIRMED opening: "…subject in the country. (3) Despite (2), the Cabinet Secretary may require a data controller who processes personal data outside Kenya".
- **Regulation 2 (definitions)** — CONFIRMED that **neither "strategic interest of the state" nor "public good" is defined**. Evidence: a search for `"strategic interest"` restricted to LN 263 returns a match anchored only to `#part_IV` (Part IV, where reg. 26 sits), none to Part I/reg. 2; a search for `"public good"` returns no hit in LN 263 at all; the reg. 2 snippet begins "2. Interpretation In these Regulations, unless the context otherwise requires — Act means…". The full list of reg. 2 definitions could not be displayed, so this is an index-based (negative) confirmation.
- **s.50 DPA** — CONFIRMED (KL snippet, Cap. 411C, Part VI, "50. Processing through a data server or data centre in Kenya"): "The Cabinet Secretary may prescribe, based on grounds of strategic interests of the state ... of revenue, certain nature of processing that shall only be effected through a server". Note the Act says "strategic interests" (plural); the Regulation says "strategic interest".
- URLs read: `https://kenyalaw.org/search/api/documents/?search=...` (results in `src/klq_servingcopy.json`, `klq_reg26_2.json`, `klq_reg26_b.json`, `klq_reg26_c.json`, `klq_reg26_e.json`, `klq_reg26_f.json`, `klq_reg26_h.json`, `klq_reg2_def.json`, `klq_ln263_publicgood.json`, `klq_reg26_a.json`); canonical document `https://kenyalaw.org/akn/ke/act/ln/2021/263/eng@2022-12-31` (WAF-challenged).

### A2. ODPC Guidance Notes for Cross-border Data Transfers (2026) — STILL UNVERIFIED (blocked)
- Both URLs in the report (`.../2026/04/Guidance-Note-on-Cross-border-Data-Transfers.pdf`; `.../2026/09/ODPC-–-Guidance-Notes-for-Cross-border-Data-Transfers.pdf`) return `404` behind the ODPC maintenance page. No mirror on an allowed host was found (KICTANet carries only commentary).
- What search-engine snippets of secondary sources say (not primary; recorded only as leads): final notes published 8 Sept 2026 after an April–15 May 2026 draft; "storing personal data on cloud servers outside Kenya is a cross-border transfer"; transfer includes "transmission, access, or making available" of data to a recipient outside Kenya; standard contractual clauses (C2C and C2P), a compliance checklist and a BCR application form annexed; localisation "confined to the specific purposes listed in Regulation 26". **Sources conflict on the notification duty**: one summary says controllers "are required to notify the ODPC of the transfer", another says transfers must be recorded and notified "on request". The report's statements on reg. 26, cloud storage, notification and s.49 consent therefore remain unverified as to wording.
- A new primary pointer: *Gachoka & 3 others v Cabinet Secretary, National Treasury & 11 others* [2026] KEHC 13493 (KLR), 15 Sept 2026 (judgment on the sale of the State's Safaricom shares) describes "26 of the Data Protection (General) Regulations, 2021 (Legal Notice 263 of 2021) which ... imposes data localisation for the "strategic interests of the State", expressly including "facilitating the conduct of elections" and records that the petitioners "argued that Safaricom PLC hosts critical national systems, including the Independent [Electoral and Boundaries Commission's]…" and "that the Independent Electoral and Boundaries Commission (IEBC) relied on Safaricom ... election result transmission network" (KL snippets, `src/klq_gach_*.json`). The court's disposition of that argument was not reachable. URL: `https://kenyalaw.org/akn/ke/judgment/kehc/2026/13493/eng@2026-09-15`.

### A3. Elections (Amendment) (No. 2) Bill, 2024 (Senate Bill No. 29 of 2024) — server-location clause — CONFIRMED (press report was right)
- National Assembly Bills Tracker as at 15 Oct 2026 (`src/parl_bills_tracker_2026-10-15.pdf`, row 202): "The Elections (Amendment) (No.2) Bill, 2024 — Sen. Bill No. 29 of 2024 — Gazette 99 — 1st read 13/02/2025 — 2nd read 01/10/2026 — Committee stage: Pending".
- Hansard, 22 Oct 2026 (`src/hansard_2026-10-22.pdf`, pp. 34–35): Hon. Peter Kaluma moved "THAT, the Bill be amended by deleting clause 23 and substituting therefor the following new clause— … (4A) In procuring the technology, the Commission shall ensure that, subject to the Constitution, this Act and the Data Protection Act, the technology is accessible and available for scrutiny by candidates, courts and such other persons as required under the law." He explained the deletion of proposed paras (i) ("intellectual property rights to the technology vest in the Commission"), (ii) and "(iv) processing of data is only effected through a server or data centre located in Kenya", asking "where this place called the 'Cloud' is". Record: "(Question, that the words to be left out be left out, put and agreed to) … (Clause 23 as amended agreed to)". Later the same day: "(The Bill was accordingly read a Third Time and passed)". Because it is a Senate Bill passed with National Assembly amendments, it goes back to the Senate (Art. 112) — consistent with the report.
- URLs: `https://www.parliament.go.ke/sites/default/files/2026-10/The%20Hansard%20-%20Thursday%2C%2022%20October%202026_0.pdf`; `https://www.parliament.go.ke/sites/default/files/2026-10/BILLS%20TRACKER%20AS%20AT%20THURSDAY%2C%2015TH%20OCTOBER%202026.pdf`.

### A4. Whether IEBC systems are gazetted critical information infrastructure (CMCA s.9) — STILL UNVERIFIED
- Kenya Gazette notices are served from Kenya Law `/akn/` paths (challenged). Not checked by any other route.

---

## B. Sensitive personal data and signatures

### B1. DPA s.2 definitions — CONFIRMED
- "biometric data" (KL snippet): "biometric data" means personal data resulting from specific technical processing based on physical, physiological or behavioural characterisation"; and "...earlobe geometry, retinal scanning and voice recognition;". Matches the report.
- "sensitive personal data" (KL snippet): ""sensitive personal data" means data revealing the natural person's race, health status, ethnic social origin". Opening matches the report; the rest of the list was not displayed. No "political" term appears in the DPA snippets.
- URL: `https://kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31` (via search API; `src/klq_dpa_bio.json`, `klq_dpa_sens.json`).

### B2. Elections (Technology) Regulations 2017 (LN 68 of 2017), reg. 2 "biometric" — CONFIRMED verbatim
- From the IEBC-hosted Gazette copy (Kenya Gazette Supplement No. 61, 21 April 2017, Legislative Supplement No. 27): ""biometric" means unique identifiers or attributes including fingerprints, hand geometry, earlobe geometry, retina and iris patterns, voice waves, DNA, and signatures;". Also reg. 15(1): "The Commission shall store and classify data in accordance with the principles set out in the Access to Information Act, 2016." Reg. 17: "All electronic data relating to an election shall be retained in safe custody by the Commission for a period of three years after the results of the elections have been declared…". Note the definition is "in these Regulations" only; it does not amend the DPA's definition.
- URL: `https://www.iebc.or.ke/uploads/resources/8lJsH5aTCd.pdf` (`src/iebc_tech_regs_2017.pdf/.txt`).

### B3. Whether a handwritten signature is "biometric data" under ODPC guidance — STILL UNVERIFIED (blocked)
- No ODPC document reachable. The only primary texts are those in B1–B2: the DPA definition requires data "resulting from specific technical processing"; the Elections (Technology) Regulations list "signatures" as biometric for their own purposes. No court decision on signatures was found on Kenya Law.

### B4. Draft ODPC notice under s.47 adding "political affiliation" — CONFIRMED as to existence and draft status; text STILL UNVERIFIED
- The ODPC page `https://www.odpc.go.ke/draft-additional-categories-of-sensitive-personal-data/` (title in search index: "DRAFT - Public Notice of Prescription of Additional Categories of Sensitive Personal Data Pursuant to Section 47 of the Data Protection Act, 2019") exists but is behind the maintenance page; the PDF URL in the report returns 404.
- Secondary confirmation on a reachable host: *The Standard*, 1 Oct 2026 (`src/std_ai_oped.html`): "The ODPC is currently seeking public comments on a draft notice proposing political affiliation as an additional category of sensitive personal data in specified cross-border-transfer circumstances. The notice says political affiliation includes information from which political leanings can reasonably be inferred. The deadline for comments is September 25, 2026." Search snippets (DataGuidance) give publication date 11 Sept 2026 and add "trade union membership".
- Status: no gazetted or final notice was found on Kenya Law (search for "sensitive personal data" "political affiliation" "section 47" returns only judgments). Treat as **still a draft** as at 31 Oct 2026.
- URL read: `https://www.standardmedia.co.ke/sci-tech/article/2001559121/africa-is-building-its-ai-future-who-is-building-it-around-the-citizen`.

### B5. s.49 consent for sensitive data processed outside Kenya — report text already VERIFIED; corroborated
- Worldcoin judgment (KL snippet): "Section 48 of the Data protection Act prohibits the transfer of personal data to"; "…sought for transferring data outside Kenya." See D3.

---

## C. Publishing personal data, severing identifiers, access to information

### C1. *Orange Democratic Movement Party (ODM) v IEBC* (Petition 440 of 2019) [2019] KEHC 2492 (KLR), Makau J, 6 Nov 2019 — CORRECTED (material)
- **What the report said:** the court ordered the Kibra register released "minus personal details" (ID/passport numbers) while names were disclosed, and the memo used this as the leading authority for severing identifiers.
- **What the judgment says (KL snippets, `src/klq_odm_*.json`):**
  - IEBC's position: "It claims, that it provided redacted details of the identification numbers of the ... details of the identification numbers of the voters in order to protect the confidentiality"; "voters in order to protect the confidentiality of their personal details urging, ... of their personal details and the right of the voter to privacy of their personal".
  - ODM's position: the register "contain the names of the voters, identification numbers (truncated), polling station, number and voter registration number"; "incomplete … for all purposes and intentions".
  - Finding on privacy: "**The Respondent has not demonstrated how Article 31 of the Constitution of Kenya is** ..." [infringed].
  - Finding on IDs: "…production of I/D Card or passport and recording his name, I/D or passport number and even telephone number. ... **I find for free and fair election to be had, national I/D or passport numbers provide** ..."; "The particulars required to be supplied in Form A is the National I/D or passport".
  - Result: "In view of my findings herein above, I find and hold that this petition succeeds."; order "(c) An order of mandamus be and is **HEREBY issued compelling the Respondent to publish the Kibra Constituency voters' register containing the names, National ID and Passport numbers and their** ..." [polling stations]; costs: "...its own costs".
- **Corrected reading:** the High Court rejected IEBC's privacy objection and ordered publication of the register **with** national ID/passport numbers. The "minus personal details" line came from one press report; *The Standard* (6 Nov 2019, `src/std_kibra.html`) wrote "release the Kibra voters register safe for confidential information", and Citizen's reports conflict. ODM v IEBC is therefore authority **against**, not for, severing ID numbers from an electoral record requested by a party under Art. 35. The recommendation to mask IDs on published result forms must rest on the DPA itself (ss.25(d), 41, 51(1)), ATI s.6(1)(d), and Art. 24(1)(e), not on this case.
- URLs: `https://kenyalaw.org/akn/ke/judgment/kehc/2019/2492/eng@2019-11-06` (via search API); `https://www.standardmedia.co.ke/nairobi/article/2001348370/iebc-ordered-to-release-kibra-voters-register`; `https://www.the-star.co.ke/news/2019-10-31-odm-says-kibra-voter-register-incomplete-demands-id-numbers/`.

### C2. *Mue & another v Chairperson of IEBC & 3 others* (Presidential Election Petition 4 of 2017) [2017] KESC 45 (KLR), 11 Dec 2017 (Ruling) — citation CONFIRMED; holding partly recovered
- Kenya Law title and date confirmed (KL). Headnote snippet: "to the right of information - what was the balance between the rights to privacy ... of right of access to information (1) Pursuant to article 24 of the Constitution"; "Constitution of Kenya, 2010 articles 24(1), 31, 35 & 40; Access to…".
- Reasoning snippets: "**The information held by the State or State organs, unless for very exceptional circumstances** ..."; "We also recognize that information held by the State or State organs, unless for ..."; "...24(1) of the Constitution, further sets limitations on rights including necessarily the right to access to information"; "(3) Subject to article 35 of the Constitution, the Commission may decline ..." (quoting IEBC Act s.27); petitioners "(Njonjo Mue and Khelef Khalifa) contended that the said memos were ..." (memos by CEO Ezra Chiloba, annexure NM-3). The operative orders were not reachable. Related rulings of 14 Nov 2017 exist: [2017] KESC 28, [2017] KESC 30, [2017] KESC 10. Supreme Court website 2017 page does not list this ruling.
- URL: `https://kenyalaw.org/akn/ke/judgment/kesc/2017/45/eng@2017-12-11` (via search API; `src/klq_mue*.json`).

### C3. *Odinga & another v IEBC & 2 others* (Election Petition 1 of 2017) [2017] KESC 31 (KLR), 28 Aug 2017 (Ruling) — CONFIRMED
- KL snippet: "Orders Petitioners as well as the 3rd respondent acquired a read only ... Log in trail of users and equipments into the IEBC Servers." (`src/klq_mue2.json`.)

### C4. *Tumaz and Tumaz Enterprises Ltd & 2 others v National Council for Law Reporting* (Misc. Civil Case E144 of 2021) [2022] KEHC 14747 (KLR), 1 Nov 2022 — CORRECTED (case name supplied)
- The report had "case name not retrieved". Headnote snippet: "– law reporting - duty of the National Council for Law Reporting (Kenya Law) to ...". URL: `https://kenyalaw.org/akn/ke/judgment/kehc/2022/14747/eng@2022-11-01`.

### C5. *Katiba Institute v Presidents Delivery Unit & 3 others* (Constitutional Petition 468 of 2017) [2017] KEHC 2183 (KLR), 8 Nov 2017 — CONFIRMED (citation; holding not re-read)

### C6. *Nairobi Law Monthly Co. Ltd v KenGen & 4 others; Transparency International & another (Amicus)* (Petition 278 of 2011) — CORRECTED citation: [2013] KEHC 6054 (KLR), 13 May 2013
- URL: `https://kenyalaw.org/akn/ke/judgment/kehc/2013/6054/eng@2013-05-13`.

### C7. *Kariuki v Attorney General* (2006 petition) — partly confirmed
- Kenya Law has *Peter M. Kariuki v Attorney General* [2011] KEHC 1564 (KLR), 8 July 2011, concerning "salaries and allowances payable to Armed Forces officers of the rank of Major General from ... 1983 to date" with the State's argument that the information "regarding salaries and allowances of the petitioner and Major Generals is confidential". Holding not displayed; the "Petition 403 of 2006" number was not confirmed. URL: `https://kenyalaw.org/akn/ke/judgment/kehc/2011/1564/eng@2011-07-08`.

### C8. *T O. S v Maseno University & 3 others* [2016] KEHC 6436 (KLR), 10 Mar 2016 — CONFIRMED citation
### C9. *COM v Standard Group Ltd & another* (Petition 192 of 2011) [2013] KEHC 6061 (KLR), 5 Apr 2013 — CONFIRMED (snippet: "Right to Privacy is enshrined in Article 31 which provides that; "Every person…").
### C10. *Khalifa* (SGR contracts) — CORRECTED: *Attorney General v Khalifa & 3 others* (Civil Appeal E085 of 2022) [2026] KECA 913 (KLR), 15 May 2026
- The report had "reported 2024–25". Snippets: "in accordance with Article 24 of the Constitution, which provides the standards ..."; "...35(1)(a) and also to publicise the information in accordance with Article 35(3)". Outcome not displayed. URL: `https://kenyalaw.org/akn/ke/judgment/keca/2026/913/eng@2026-05-15`.
### C11. *KHRC v Communications Authority & 4 others* (Const. Petition 86 of 2017) [2018] KEHC 7494 (KLR), 19 Apr 2018 — CONFIRMED; Court of Appeal is *Communications Authority of Kenya v Okoiti & 8 others* (Civil Appeal 166 & 167 of 2018 (Consolidated)) [2020] KECA 754 (KLR), 24 Apr 2020 — CORRECTED party name (report: "v Omtatah").
### C12. *Free Kenya Initiative & 17 others v IEBC & 7 others* (Const. Petition E160 of 2022 & E219, E225 & 12 of 2022 (Consolidated)) [2022] KEHC 10217 (KLR), 5 July 2022 — CORRECTED (citation found for the "supporters' ID copies" case)
- Snippet: "...Protection Act and the Registration of the Persons Act and therefore null and void". *Lichete v IEBC* [2022] KEHC 13244 (18 July 2022) confirms: "Given that this Court voided the requirement to supply copies of identity cards of…". Also *Rajput & another v IEBC* [2022] KEHC 17164 (16 Dec 2022) contains the line "45 of the Data Protection Act, National Identity Cards qualify as sensitive data" (context — a party's submission or the court's — not displayed).
### C13. 2026 High Court decision on party agents'/observers' access — CORRECTED (citation found): *Okoiti & another v IEBC & 5 others* (Const. Petition E757 & E753 of 2025 (Consolidated)) [2026] KEHC 13916 (KLR), Mutai J, 22 Sept 2026
- Snippets: heading "(iii) Party agent and observer access."; "Candidates' agents and accredited observers are entitled to be present at, and to ..."; quotes s.39(1C)(c) "the polling result forms on an online public portal maintained by the Commission"; "Final Orders 83." (orders not displayed).
### C14. *KHRC & another v Chebukati & 2 others* (Petition E400 of 2022) [2025] KEHC 8 (KLR), 9 Jan 2025 — CORRECTED (the report said "outcome not found"; a judgment exists)
- Snippet: "On the portal, one could access all Form 34As, 34Bs and 34C." Disposition not displayed.
### C15. *Odinga & 16 others v Ruto & 10 others* [2022] KESC 54 (5 Sept 2022) and [2022] KESC 56 (26 Sept 2022) — CONFIRMED (snippet: "between Forms 34A uploaded on the IEBC Public Portal and the Forms 34A received…"). *Youth Advocacy for Africa & 7 others v IEBC* [2022] KESC 42 (30 Aug 2022, scrutiny ruling) also confirmed; the Supreme Court website serves its PDF (`src/sc_att_4703.bin`).
### C16. *Ndung'u & another v Wachira & another* (Const. Petition E047 of 2025) [2025] KEHC 7265 (KLR), 26 May 2025 (Ruling) — CONFIRMED exists (interlocutory ruling; merits not shown).
### C17. *Anne Waiguru v Google Inc & 2 others* [2014] KEHC 8498 (KLR), 5 Dec 2014 — CONFIRMED citation.
### C18. *Odhiambo & another v Attorney General & another* (Petition E290 of 2024) [2026] KEHC 3809 (KLR), 19 Mar 2026 — CONFIRMED citation (report had none)
- Snippets: "phone number forms part of his/her digital identity just like the physical national ..."; "The Petition challenges the reassignment and/or recycling of the deactivated but ..."; "...privacy under Article 31 and the right to communication". The six-month order reported by the press was not displayed.
### C19. ODPC caution on circulating MPs' phone numbers (2026) — STILL UNVERIFIED (no primary or allowed-host report found).
### C20. Access to Information (General) Regulations 2023 (severance) — STILL UNVERIFIED (not located through the search API).
### C21. Registration of Persons Act non-disclosure rule — STILL UNVERIFIED (not re-checked; Kenya Law page challenged).

---

## D. Data protection impact assessment

### D1. DPA s.31(5) — CONFIRMED verbatim (KL snippet): "(5) The data impact assessment reports shall be submitted sixty days prior to the processing" (continues "of data. (6) The Data Commissioner shall set out guidelin[es]"). Also reproduced in Kenya Law Bench Bulletin Issue 55 (1 Oct 2021).
### D2. Regulation 49 — CONFIRMED (heading "49. Processing activities requiring data protection impact assessment"; snippets): "rights and freedoms of a data subject include — (a) automated decision making with legal or similar significant effect"; "significant effects; (b) use of personal data on a large-scale for a purpose other than that for which the data was initially collected"; "(c) processing biometric or genetic data ... personal data or data relating to children or vulnerable groups; (f) combining, linking or cross-referencing"; "systematic monitoring of a publicly accessible area on a large scale; (i) innovative use ... prior to processing data under subregulation (1) conduct a data protection impact assessment".
### D3. *Republic v Tools for Humanity Corporation (US) & 9 others; Katiba Institute & 4 others (Ex parte Applicants)* (Judicial Review Application E119 of 2023) [2025] KEHC 5629 (KLR), Aburili J, 5 May 2025 — CONFIRMED (title corrected: "& 9 others", not "& 8 others"; the "Data Privacy & Governance Society of Kenya (Interested Party)" caption belongs to the later ruling [2025] KEHC 16829 of 14 Nov 2025)
- Orders (KL snippets): "(d) Mandamus compelling the 1st to 5th Respondents to, within 7 days of the order ... 7 days of this order, permanently erase and destroy (under the supervision of the [Data Commissioner]) ..."; "Data Protection Impact Assessment contrary to section 31 of the Data Protection [Act]"; "Section 48 of the Data protection Act prohibits the transfer of personal data to ..."; "as data controllers or processors in Kenya as required under section 18 of the Data [Protection Act]"; and the court's remark that the "Protection Act, 2019 requires amendments to align with emerging challenges and technological…". Katiba Institute's statement (`src/katiba_wc1.html`) describes orders of prohibition, certiorari and mandamus (7-day deletion under the Data Commissioner's supervision); ODPC confirmed deletion on 20 Jan 2026 (Katiba/Star repost, `src/katiba_wc2.html`). Related: *ODPC v Tools for Humanity (Worldcoin) & 2 others* (Misc. Criminal Application E315 of 2023) [2024] KEHC 312 (25 Jan 2024).
- URL: `https://kenyalaw.org/akn/ke/judgment/kehc/2025/5629/eng@2025-05-05`.
### D4. *Republic v Mucheru, CS ICT & 2 others; Katiba Institute & another (Ex parte); Immaculate Kasait, Data Commissioner (Interested Party)* (JR Application E1138 of 2020) [2021] KEHC 122 (KLR), Ngaah J, 14 Oct 2021 — CONFIRMED
- Snippets: "the decision ... 2020 to roll out Huduma Cards for being ultra vires section 31 of the Data Protection [Act]"; "(c) Mandamus compelling the respondents to conduct a data protection impact assessment"; "...Protection Act came into force and that for this reason, section 31 of the Data …" (retrospectivity argument). Court of Appeal: *Mucheru & 2 others v Katiba Institute & 2 others* (Civil Application E373 of 2021) [2022] KECA 386 (KLR), 4 Mar 2022 (Ruling) — CONFIRMED.
### D5. *Nubian Rights Forum & 2 others v Attorney General & 15 others* (Petition 56, 58 & 59 of 2019 (Consolidated)) — CORRECTED: neutral citation [2020] KEHC 8772 (KLR), 30 Jan 2020 (the report said none found).
### D6. *Haki na Sheria Initiative v Attorney General & 4 others* (Petition 196 of 2023) [2024] KEHC 10021 (KLR), 12 Aug 2024 (Ruling) — CONFIRMED.
### D7. ODPC Guidance Note on DPIA (28 Nov 2023) — STILL UNVERIFIED (blocked).

---

## E. Breach notification

### E1. Regulation 37 (heading "Categories of notifiable data breach") — CONFIRMED; the report's bracketed reconstruction was right
- Snippets: "the purpose of section 43 of the Act, a data breach is taken to result in real risk of harm ... data or classes of personal data relating to the data subject set out in the Second Schedule"; "to a data subject if that data breach relates to— (a) the data subject's full name ... or identification number and any of the personal data or classes of personal [data]"; "..., security code, access code, response to a security question, biometric data or ..."; "data referred to in sub-paragraph (3)(a) shall not be publicly available solely because of any data breach".
### E2. LN 264 of 2021, reg. 14 ("Outcome of investigation") — CONFIRMED: "(1) Data Commissioner shall, upon the conclusion of the investigations, make a determination ... based on the findings of the investigations. (2) A determination under subregulation…"; "(d) recommendation for prosecution; or **(e) an order for compensation to the data subject by the respondent**". The report's "reg. 14(3)(e)" attribution is consistent (sub-regulation number inferred from the lettering; the "(3)" itself was not displayed). URL: `https://kenyalaw.org/akn/ke/act/ln/2021/264/eng@2022-12-31` (via search API).

---

## F. Registration of the IEBC with the ODPC

### F1. ODPC register entry — STILL UNVERIFIED (blocked: ODPC site under maintenance; no register page reachable).
### F2. New primary evidence bearing on it — National Assembly Hansard, Thursday 15 Oct 2026, pp. 31–33 (`src/hansard_2026-10-15.pdf`)
- Motion (Hon. Umulkher Harun, Vice-Chair, Constitutional Oversight Committee) "THAT, this House adopts the Fourth Report of the Constitutional Oversight Committee, laid on the Table of the House on Wednesday, 19th November 2025, on the Status of Implementation of the Constitution by – (i) The Independent Electoral and Boundaries Commission; and (ii) The Office of the Data Protection Commissioner." Recommendations read into the record include: "5. All registered political parties that have not registered with the ODPC to complete registration. **IEBC to expedite its registration and full compliance, including data processing and sharing agreements, before the 2027 General Election.** 6. The ODPC to update its Guidance Note on Processing of Personal Data for Electoral Purposes within six months to address emerging technologies, and to invest in digital forensic and real-time monitoring tools." Seconder: "expedited registration and full compliance by political parties and the IEBC with ODPC requirement ahead of 2027". Record: "(Question put and agreed to)" — the motion was adopted.
- Reading: as at the Committee's engagement (Sept–Oct 2025) and the House's adoption (15 Oct 2026), Parliament treated the IEBC's ODPC registration as **not complete**. This also CONFIRMS the report's "October 2026 National Assembly resolution" item (§5.1).
- URL: `https://www.parliament.go.ke/sites/default/files/2026-10/The%20Hansard%20-%20Thursday%2C%2015%20October%202026.pdf`.
### F3. ODPC FAQ list of non-exempt sectors — STILL UNVERIFIED (blocked).

---

## G. Data Protection (Amendment) Bill 2025 / any 2024–2026 amendment

### G1. — CORRECTED: no such Bill is before Parliament, and no amending Act exists
- National Assembly Bills Tracker 2026 ("overview of the status of all Bills before the National Assembly in the year 2026", 80 pp., as at 15 Oct 2026): no "Data Protection" entry (the only "Protection" hits are the Whistleblower Protection Bill 2023, Care and Protection of Child Parents Bill 2023, Social Protection Bill 2025, Plant Protection Bill 2025, Street Vendors (Protection of Livelihood) Bill 2023).
- National Assembly bills pages (all 15 pages for the 2022– Parliament) and Senate bills page: no "Data Protection" entry. Parliament library (DSpace API, query "Data Protection (Amendment) Bill"): one unrelated hit.
- Kenya Law bills collection (which carries National Assembly bills, e.g. the KICA (Amendment) Bill 2025): query `"Data Protection (Amendment) Bill"` → 0 documents. Kenya Law's consolidated DPA expression remains `eng@2022-12-31`; a query for `"Data Protection Act, 2019 is amended"` returns no Act.
- Secondary snippets describe the "Data Protection (Amendment) Bill, 2025" as a draft "developed by the Data Privacy and Governance Society of Kenya (DPGSK)" (civil-society text). The aggregator claim that it "took effect 5 Oct 2026" is unsupported by any primary source.
- URLs: tracker and bills pages above; `https://kenyalaw.org/search/api/documents/?search=%22Data+Protection+(Amendment)+Bill%22` (`src/klq_dpabill.json`).

---

## H. ODPC guidance notes (electoral, DPIA, cross-border, AI, public sector, others) — STILL UNVERIFIED (blocked)
- All `odpc.go.ke` URLs return 404/maintenance. No copies found on `kictanet.or.ke`, `katibainstitute.org`, `khrc.or.ke` or the news sites. The electoral Guidance Note's date, and any passage on result forms, agents, polling officials, publication, retention or security, remain unread. Note the Kenya Law corpus (judgments 2023–2026) does not quote the electoral Guidance Note either (no hit for "Guidance Note" + "Electoral Purposes" in judgments).

---

## I. ODPC determinations and penalties

| Item | Outcome | Evidence (verbatim) | URL read |
|---|---|---|---|
| Oppo / Eagleage Ltd — KES 5,000,000 | CONFIRMED | *Republic v Data Protection Commissioner; Eagleage Limited t/a Oppo Aed Kenya (Exparte)* (Misc. Application E005 of 2023) [2023] KEHC 20418 (KLR), 14 July 2023 (Ruling): "...Protection Commissioner imposing a penalty of Kshs 5,000,000/- on the exparte [applicant]" | `https://kenyalaw.org/akn/ke/judgment/kehc/2023/20418/eng@2023-07-14` (KL) |
| Whitepath Co. Ltd — amount | CORRECTED: KES 5,000,000 | *The Star*, 12 Apr 2023: "fined Whitepath Company Limited Sh5 million … failed to comply with the enforcement notice dated January 10, 2023 … close to 150 complaints … Whitepath will have to pay the ODPC a penalty of Sh5 million pursuant to Section 63 of the Data Protection Act, and Regulation 20 of the Data Protection (Complaints Handling Procedure and Enforcement)" | `https://www.the-star.co.ke/sports/football/2023-04-12-digital-credit-provider-fined-sh5m-for-personal-data-breach` |
| Regus Kenya — KES 5m, reduced to 2.5m | CONFIRMED; citation CORRECTED | Star 12 Apr 2023: "Regus Kenya was non-cooperative and failed to respond to a Notification of Complaint dated October 27, 2022 … frequent spamming". *Regus Kenya Limited v Data Protection Commissioner & another* (Civil Appeal **E472 of 2023**) [2025] KEHC 13491 (KLR), A.C. Mrima J, 30 Sept 2025: appeal "from the decision of the Data Protection Commissioner dated 23rd May 2023 and Penalty [Notice]"; "The parties will now meet midway as the sentence is reviewed to Kshs. 2,500,000/=". A further ruling [2026] KEHC 12613 (30 July 2026). The report's "Civil Appeal E47" and "ODPC decision 16 Feb 2023" are not borne out by the judgment title/snippet. | `https://kenyalaw.org/akn/ke/judgment/kehc/2025/13491/eng@2025-09-30` (KL) |
| Mulla Pride — KES 2,975,000; Casa Vera — KES 1,850,000; Roma School — KES 4,550,000 (26 Sept 2023) | CONFIRMED | *The Star*, 26 Sept 2023, quoting the ODPC statement: "Mulla Pride Ltd … penalty of Sh2,975,000 … using names and contact information of the complainants, which were obtained from third parties"; "Casa Vera Lounge … fined Sh1,850,000 for posting a reveller's image"; "Roma School … fined Sh4,550,000 for posting minors' pictures without parental consent"; "issued pursuant to section 62 and 63 of the Data Protection Act, 2019 (Act) and Regulation 20 and 21 of the Data Protection (Complaints Handling Procedure and Enforcement) Regulations, 2021". *The Standard*, 26 Sept 2023, same figures. | `https://www.the-star.co.ke/news/2023-09-26-mobile-lender-fined-millions-for-breach-of-borrowers-privacy`; `https://www.standardmedia.co.ke/national/article/2001482275/roma-school-fined-sh45m-for-posting-minors-image-without-parental-consent` |
| Roma School judicial review | CONFIRMED citation | *Roma School Uthiru v Data Protection Commissioner; Muturi (Interested Party)* (JR E165 of 2023) [2024] KEHC 3927 (KLR), 3 Apr 2024 (Judgment); complaint "ODPC/CONF/1/7/3 VOL 1(7) Christine Muturi v Roma School Uthiru". Electronic-service holding not displayed. | KL |
| Platinum Credit — KES 400,000 compensation | Partly confirmed | Nation snippet (host blocked): "Samuel Kamau Vs Platnum Credit Award: Sh400,000". Appeal exists: *Platinum Credit Limited v Data Protection Commissioner & another* (Civil Appeal E282 of 2025) [2026] KEHC 7257 (KLR), 25 May 2026 — "complaint lodged by the 2nd Respondent, Samuel Kamau Waweru"; ODPC "submissions opposing the Appellant's appeal". Outcome not displayed. Also [2026] KEHC 6202 (7 May 2026) and *Platinum Credit Ltd v Ngolo & another* [2026] KEHC 12912 (18 Aug 2026). | KL |
| Bliss Park Hotel (ID card on Facebook) | STILL UNVERIFIED | No hit on Kenya Law ("Bliss Park" → 0), Star or Standard; ODPC site down. | — |
| DTB, Nairobi Hospital, Liquid Telecom, Fin Africa, Eldoret Hospital, Tuma Cash, Waithira, Merceline | STILL UNVERIFIED | ODPC PDFs unreachable; Nation blocked; no Star/Standard hits. | — |
| Allen Waiyaki Gichuhi v Mathenge (public-domain documents) | STILL UNVERIFIED | ODPC/CIPIT hosts unreachable. | — |
| Worldcoin enforcement | CONFIRMED | See D3. | — |
| Political-party involuntary registration; bulk SMS; IEBC | STILL UNVERIFIED / none found | No determination on Kenya Law. Press context: *The Star*, 18 Aug 2026, "Political SMS raise questions over Kenyans' personal data" (search snippet). | — |
| Aggregate counts (215 determinations etc.) | STILL UNVERIFIED | ODPC Data Bulletin unreachable. | — |
| New: *Kenya Power & Lighting PLC v Chigiti* (Civil Appeal E723 of 2025) [2026] KEHC 14569, 29 Sept 2026 | Note | On appeals from ODPC "enforcement and penalty notices"; cites *Regus*. | KL |

---

## J. Other items

### J1. Elections (General) Regulations 2012, reg. 79 and Form 34A — CONFIRMED (follow-up 5 done)
- From the IEBC-hosted "[Rev. 2017]" compilation of Elections Act subsidiary legislation (`src/iebc_elections_subsidiary.pdf`, pp. 109–110, 145–146): reg. 79(1) "The presiding officer, the candidates or agents shall sign the declaration in respect of the elections." 79(2)(a) "Presidential election results shall be in Form 34A set out in the Schedule"; 79(2A)(b) "request each of the candidates or agents present to append his or her signature"; (d) "affix a copy of the declaration of the results at the public entrance to the polling station…"; 79(3)–(7) on refusal/absence. Form 34A [Reg. 79(2)(a)] declaration block: "Presiding Officer … Signature … Date"; "Deputy Presiding Officer … Signature … Date"; table "Agents or Candidates (if present) — No. | Name of Candidate or Agent | ID/Passport No. | Party Name/Independent Candidate | Tel. Contact | Signature | Date"; "Reasons for Refusal to Sign (if any)". Form 34B [Reg. 87(1)(a)] carries the same agents table plus "Name of the Constituency Returning Officer … ID Number … Signature … Date". Kenya Law also lists LN 128 of 2012 (consolidated `eng@2022-12-31`) with the heading "Candidates, etc. to sign declaration".
- URL: `https://www.iebc.or.ke/uploads/resources/902SFZiDqt.pdf`.
### J2. KICA SIM-registration regulations — note: Kenya Law lists **The Kenya Information and Communications (Registration of Telecommunications Service Subscribers) Regulations, LN 90 of 2025** (30 May 2025). The report cites the 2015 Regulations (LN 163 of 2015); whether LN 90 of 2025 replaces them was not checked.
### J3. KICTANet / KHRC — no copies of ODPC documents; KICTANet carries only commentary ("What does the Data Protection Act say about cross border data transfers", c. 2023).
### J4. Supreme Court website — serves the 2022 scrutiny rulings (*Youth Advocacy for Africa v IEBC*, 18 pp.) but not the 2017 Mue ruling.

---

## K. Summary of label changes applied to A-data-protection.md

- CORRECTED (substantive): ODM v IEBC (C1) — reverses the report's use of the case; Data Protection (Amendment) Bill 2025 (G1) — no bill before Parliament; Whitepath penalty amount (I); Regus appeal citation/decision date (I); Nubian Rights Forum citation (D5); Tumaz case name (C4); Nairobi Law Monthly citation (C6); Khalifa CoA citation (C10); CA v Okoiti party name (C11); Worldcoin title (D3); reg. 26 heading (A1); Free Kenya Initiative, Okoiti 2026, KHRC v Chebukati, Odhiambo citations (C12–C14, C18).
- CONFIRMED: reg. 26(1)–(3) wording and absence of reg. 2 definitions; s.50; s.2 definitions; s.31(5); reg. 37; reg. 49; LN 264 reg. 14(e); Elections (Technology) Regs reg. 2 and 15, 17; Elections (General) Regs reg. 79 and Form 34A; Huduma, Worldcoin, Roma, Oppo, Mulla Pride, Casa Vera, Regus penalties; Elections (Amendment) (No. 2) Bill clause 23 history; 15 Oct 2026 NA motion; existence/draft status of the s.47 notice; remaining case citations listed above.
- STILL UNVERIFIED (blocked or not found): every ODPC document (guidance notes, register, determinations texts, draft s.47 notice text, FAQ, Data Bulletin); IEBC register entry; CII gazettement; Bliss Park and the other 2025–26 determinations; Mue ruling disposition; ATI Regulations 2023; Registration of Persons Act; MPs' phone-number caution; Kariuki 2006 petition number.
