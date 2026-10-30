# Verifying a published result form

This guide is for anyone who wants to check, without trusting the publisher,
that a published (redacted) result form shows **exactly** the results that
were transmitted. That includes party agents, observers, journalists, courts
and auditors.

## What is claimed

For every published form, the publisher claims:

1. **Same source.** It was made from a specific original transmission,
   identified by its SHA-256 hash. You can compare that hash with the copy on
   the IEBC portal, or with the KIEMS transmission.
2. **Nothing else changed.** The only change is that some 8×8-pixel blocks were
   replaced with solid black. Every other block of the image is the same data
   as in the original.
3. **Results and stamps untouched.** None of those black blocks touches the
   results (header and QR code, station names and codes, the votes table, the
   polling-station counts) or any official stamp on the form.
4. **Fixed record.** The form, its manifest and its place in the batch were
   recorded when published, and cannot be changed later without detection.

A black block can hide a digit, but it cannot turn one digit into another. So
if claims 1–3 hold, no result on the form can have been altered. At most, the
listed blocks were hidden, and the check shows that none of them is in the
results.

## What you need

| Item | Where it comes from |
|---|---|
| The published PDF, e.g. `1_34_A_..._132155.pdf` | the publication site |
| Its manifest, `..._132155.manifest.json` | published alongside the PDF |
| The publisher's public key, `signing_key.pub.pem` | the publisher's website, announced once |
| The ledger, `ledger.jsonl` | published alongside the forms |
| The original transmission, e.g. `1_34_A_..._132155.PDF` | the IEBC portal, or the KIEMS/IEBC records |
| The publisher's certificate, `signer.crt` (or its SHA-256 fingerprint) | the publisher's website, announced once; optional |

The tool is open source. Use the exact version named in the manifest
(`tool.commit`), or review the code that does the checking: `proof.rs` and
`verify.rs` in `rust/src`, and `rust/csrc/jdct.c`, which reads the JPEG
coefficients with libjpeg.

## Run the check

```bash
sudo rust/scripts/setup-deps.sh && (cd rust && cargo build --release)   # see README.md
rfpii verify 1_34_A_..._132155.pdf \
    --original 1_34_A_..._132155.PDF \
    --pubkey signing_key.pub.pem \
    --ledger ledger.jsonl \
    --overlay changes/ \
    --trust signer.crt
```

Example output (a real Form 34A from the test set):

```
PASS  manifest signature valid
PASS  ledger hash chain intact
PASS  manifest recorded in ledger (entry 3)
PASS  redacted PDF matches the manifest's SHA-256
PASS  digital signature valid and covering the whole file (signed by C=KE, CN=Returning Officer, ... at D:20220812...)
PASS  the signing certificate is the one the manifest names
PASS  the signing certificate is trusted (--trust)
PASS  1 page image(s), as recorded
PASS  original matches the manifest's source SHA-256 85d2fa2da8b6b42d...
PASS  original has the same number of pages
PASS  page 1: image matches the manifest
PASS  page 1: no blacked block touches the 4 protected results region(s) and 1 stamp(s)
PASS  page 1: original page is a JPEG, as recorded
PASS  page 1: 57601 of 60651 blocks identical to the original (DCT coefficients); 3050 changed, all listed and solid black
PASS  page 1: results regions and stamps decode to identical pixels in original and redacted (luminance)
RESULT: VERIFIED - only the listed blocks were blacked out; nothing else changed
```

A scanned batch (one PDF of many forms) may be published page by page, as
`NAME.p012.pdf`. Its manifest names the whole original file, by its SHA-256,
and which page of it this is (`source.page`, of `source.pages`). Verify it
against the whole original, exactly as above: the check reads

```
PASS  original matches the manifest's source SHA-256 71af42cf2eb67437...
PASS  original has 10 pages, as recorded; this is page 4
PASS  page 4: 135704 of 136400 blocks identical to the original (DCT coefficients); 696 changed, all listed and solid black
```

and compares the published page with that page of the original only.

`changes/` then contains an image of the original with every changed block
shaded red, the results regions outlined in green and the stamps in magenta,
so you can see for yourself where the changes are.

The protected regions come from the manifest. For a Form 34A or 34B they are
the template's results regions; for older forms read without a template, the
whole page above the "Declaration" label. Stamps are found on each page and
listed with their circle. Whether the list is complete is something you can
check by eye on the overlay: every stamp on the form should be outlined, and
no red block should touch one.

## How the check works, and why it doesn't rely on trusting us

- **Comparing the stored data, not what a viewer shows.** KIEMS photos are JPEG
  images. A JPEG stores each 8×8 block as a set of numbers (quantised DCT
  coefficients). The verifier reads those numbers from the original and from
  the published file and compares them block by block. A block passes if it is
  identical, or if it is listed in the manifest and holds exactly the values
  for solid black. No image decoder is involved, so differences between
  viewers or libraries can't hide or fake anything. The quantisation tables
  must also be identical, so the whole image can't be re-compressed.
- **Results checked again in pixels.** The pixels inside each results region
  are also decoded from both files and compared directly.
- **Shown the same way.** Each published page shows its image as the
  original does: same page size, and turned or mirrored the same way (a
  photo's EXIF orientation, or how the original PDF places and rotates its
  image). One exception: a photo with no orientation tag that Clef judged to
  be on its side (`redact --orient`) is shown turned upright, by the
  quarter turns its manifest records. The image itself is not turned (the
  page's `/Rotate` is), and the verifier accepts a turn only for such a
  photo.
- **Signature.** The manifest is signed with the publisher's Ed25519 key. If
  anyone edits it afterwards, the signature fails.
- **The PDF's own signature.** The published PDF can also carry a digital
  signature (PKCS#7, as PDF readers check it) over the whole file, made with
  the publisher's certificate, which the manifest names by its SHA-256. Any
  PDF reader shows whether it is intact; `rfpii verify` checks it too, and
  with `--trust` that the certificate is the one you expect. A self-signed
  certificate is only as good as your knowledge of its fingerprint: compare
  it with the one the publisher announced.
- **Ledger.** Each entry in the ledger includes the hash of the previous entry.
  Changing, removing or back-dating one entry breaks every entry after it.
- **Batch root.** Each batch ends with one hash (the Merkle root) over all its
  manifests, which the publisher announces publicly when the batch is
  released. A manifest that wasn't in the batch can't be added later.
- **Re-running the redaction.** `--rerun templates/` repeats the whole redaction
  from the recorded random seed, with the recorded code and template versions
  and the recorded settings (`settings` in the manifest: the masking rules,
  and under `align` every alignment threshold, such as `--min-inliers` and
  `--min-score`; a manifest without them was made with the defaults).
  It must give a bit-identical result, which shows the published file is
  exactly what the documented process produces. For forms read without a
  template, and Forms 34A (whose comments are read with OCR), the manifest
  also records the OCR models' hashes; the re-run
  checks that the same models are installed (`setup-deps.sh` puts them in
  `/opt/rfpii/models`). Where a vision review masked more (`vlm_found` in the
  manifest) or hid a column whole (`whole_columns`), the re-run masks the
  same from the manifest: it needs no access to any model, and the review's
  own record (`vlm_review`) shows what each model was shown and answered.

Try a tampered file. Change one number in the votes table of the published
image (even one coefficient of one block), re-save it, and run `rfpii verify`.
It fails four separate checks. The test suite does this automatically
(`rust/tests/e2e.rs`).

## What the check does not prove

- **The original itself.** The check proves the published form matches the
  original you provide. Whether that original is the genuine KIEMS
  transmission is a question for the IEBC's own records: the portal copy, the
  kit logs, and the kit serial and capture time in the file name, which the
  manifest also records.
- **What is under the black blocks.** That is the point of redaction. The
  blocks are only on signatures, IDs and phone numbers, and the overlay shows
  where they are.
