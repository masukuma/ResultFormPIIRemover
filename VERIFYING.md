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
3. **Results untouched.** None of those black blocks touches the results:
   header and QR code, station names and codes, the votes table, or the
   polling-station counts.
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

The tool is open source. Use the exact version named in the manifest
(`tool.commit`), or review the ~200 lines of `proof.py` and `verify.py` that
do the checking.

## Run the check

```bash
pip install -e .
rfpii verify 1_34_A_..._132155.pdf \
    --original 1_34_A_..._132155.PDF \
    --pubkey signing_key.pub.pem \
    --ledger ledger.jsonl \
    --overlay changes/
```

Example output (a real Form 34A from the test set):

```
PASS  manifest signature valid
PASS  ledger hash chain intact
PASS  manifest recorded in ledger (entry 3)
PASS  redacted PDF matches the manifest's SHA-256
PASS  1 page image(s), as recorded
PASS  original matches the manifest's source SHA-256 85d2fa2da8b6b42d...
PASS  original has the same number of pages
PASS  page 1: image matches the manifest
PASS  page 1: no blacked block touches the 4 protected results region(s)
PASS  page 1: original page is a JPEG, as recorded
PASS  page 1: 57601 of 60651 blocks identical to the original (DCT coefficients); 3050 changed, all listed and solid black
PASS  page 1: results regions decode to identical pixels in original and redacted (luminance)
RESULT: VERIFIED - only the listed blocks were blacked out; nothing else changed
```

`changes/` then contains an image of the original with every changed block
shaded red and the results regions outlined in green, so you can see for
yourself where the changes are.

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
- **Signature.** The manifest is signed with the publisher's Ed25519 key. If
  anyone edits it afterwards, the signature fails.
- **Ledger.** Each entry in the ledger includes the hash of the previous entry.
  Changing, removing or back-dating one entry breaks every entry after it.
- **Batch root.** Each batch ends with one hash (the Merkle root) over all its
  manifests, which the publisher announces publicly when the batch is
  released. A manifest that wasn't in the batch can't be added later.
- **Re-running the redaction.** `--rerun templates/` repeats the whole redaction
  from the recorded random seed, with the recorded code and template versions.
  It must give a bit-identical result, which shows the published file is
  exactly what the documented process produces.

Try a tampered file. Change one number in the votes table of the published
image (even one coefficient of one block), re-save it, and run `rfpii verify`.
It fails four separate checks. The test suite does this automatically
(`tests/test_verify_e2e.py`).

## What the check does not prove

- **The original itself.** The check proves the published form matches the
  original you provide. Whether that original is the genuine KIEMS
  transmission is a question for the IEBC's own records: the portal copy, the
  kit logs, and the kit serial and capture time in the file name, which the
  manifest also records.
- **What is under the black blocks.** That is the point of redaction. The
  blocks are only in the signature boxes and the ID and phone columns, and the
  overlay shows where they are.
