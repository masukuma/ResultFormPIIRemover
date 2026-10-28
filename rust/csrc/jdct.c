/* JPEG at coefficient level, through libjpeg(-turbo).
 *
 * jdct_read     the quantised DCT coefficients, tables and geometry of a JPEG
 * jdct_redact   a copy of a JPEG with listed luma blocks set to black and
 *               listed chroma blocks set to neutral; every other coefficient,
 *               the quantisation tables and the sampling are copied
 *
 * libjpeg reports errors by calling error_exit, which would end the process:
 * here it longjmps back and the call returns 0 with the message in err.
 */
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

typedef struct {
    int width, height, num_components, progressive;
    int h_samp[4], v_samp[4];
    int bw[4], bh[4];       /* component size in blocks (not padded to MCUs) */
    int qtno[4];            /* quantisation table slot of each component */
    unsigned short qt[4][64]; /* table of each slot, natural order (0 if unused) */
    short *coef[4];         /* bh*bw blocks of 64 coefficients, natural order */
} jdct;

struct err_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf jump;
    char *msg;
    int msglen;
};

static void on_error(j_common_ptr cinfo) {
    struct err_mgr *e = (struct err_mgr *)cinfo->err;
    if (e->msg && e->msglen > 0) {
        char buf[JMSG_LENGTH_MAX];
        (*cinfo->err->format_message)(cinfo, buf);
        snprintf(e->msg, e->msglen, "%s", buf);
    }
    longjmp(e->jump, 1);
}

static void quiet(j_common_ptr cinfo, int level) { (void)cinfo; (void)level; }

void jdct_free(jdct *d) {
    for (int c = 0; c < 4; c++) {
        free(d->coef[c]);
        d->coef[c] = NULL;
    }
}

int jdct_read(const unsigned char *data, unsigned long len, jdct *out, char *err, int errlen) {
    struct jpeg_decompress_struct cinfo;
    struct err_mgr jerr;
    memset(out, 0, sizeof(*out));
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = on_error;
    jerr.pub.emit_message = quiet;
    jerr.msg = err;
    jerr.msglen = errlen;
    if (setjmp(jerr.jump)) {
        jpeg_destroy_decompress(&cinfo);
        jdct_free(out);
        return 0;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, len);
    jpeg_read_header(&cinfo, TRUE);
    jvirt_barray_ptr *arrays = jpeg_read_coefficients(&cinfo);
    out->width = cinfo.image_width;
    out->height = cinfo.image_height;
    out->num_components = cinfo.num_components > 4 ? 4 : cinfo.num_components;
    out->progressive = cinfo.progressive_mode ? 1 : 0;
    for (int s = 0; s < NUM_QUANT_TBLS && s < 4; s++) {
        if (cinfo.quant_tbl_ptrs[s])
            for (int i = 0; i < 64; i++)
                out->qt[s][i] = cinfo.quant_tbl_ptrs[s]->quantval[i];
    }
    for (int c = 0; c < out->num_components; c++) {
        jpeg_component_info *ci = cinfo.comp_info + c;
        out->h_samp[c] = ci->h_samp_factor;
        out->v_samp[c] = ci->v_samp_factor;
        out->bw[c] = ci->width_in_blocks;
        out->bh[c] = ci->height_in_blocks;
        out->qtno[c] = ci->quant_tbl_no;
        size_t n = (size_t)ci->width_in_blocks * ci->height_in_blocks * 64;
        out->coef[c] = (short *)malloc(n * sizeof(short));
        if (!out->coef[c]) {
            snprintf(err, errlen, "out of memory");
            jpeg_destroy_decompress(&cinfo);
            jdct_free(out);
            return 0;
        }
        for (JDIMENSION r = 0; r < ci->height_in_blocks; r++) {
            JBLOCKARRAY row = (cinfo.mem->access_virt_barray)((j_common_ptr)&cinfo, arrays[c], r, 1, FALSE);
            memcpy(out->coef[c] + (size_t)r * ci->width_in_blocks * 64, row[0],
                   (size_t)ci->width_in_blocks * 64 * sizeof(short));
        }
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return 1;
}

static int keep_marker(jpeg_saved_marker_ptr m) {
    /* Markers that affect how colours decode (ICC profile, Adobe); EXIF, XMP
       and comments can carry camera serials or GPS and are dropped. The JFIF
       header (APP0) is written by libjpeg itself. */
    return m->marker == JPEG_APP0 + 2 || m->marker == JPEG_APP0 + 14;
}

int jdct_redact(const unsigned char *data, unsigned long len,
                const unsigned char *luma, int luma_rows, int luma_cols,
                const unsigned char *chroma, int chroma_rows, int chroma_cols,
                unsigned char **out, unsigned long *outlen, char *err, int errlen) {
    struct jpeg_decompress_struct src;
    struct jpeg_compress_struct dst;
    struct err_mgr jerr; /* one handler for both: an error in either jumps here */
    *out = NULL;
    *outlen = 0;
    src.err = jpeg_std_error(&jerr.pub);
    dst.err = &jerr.pub;
    jerr.pub.error_exit = on_error;
    jerr.pub.emit_message = quiet;
    jerr.msg = err;
    jerr.msglen = errlen;
    volatile int dst_created = 0;
    if (setjmp(jerr.jump)) {
        if (dst_created)
            jpeg_destroy_compress(&dst);
        jpeg_destroy_decompress(&src);
        if (*out) {
            free(*out);
            *out = NULL;
        }
        return 0;
    }
    jpeg_create_decompress(&src);
    jpeg_mem_src(&src, data, len);
    jpeg_save_markers(&src, JPEG_APP0 + 2, 0xFFFF);
    jpeg_save_markers(&src, JPEG_APP0 + 14, 0xFFFF);
    jpeg_read_header(&src, TRUE);
    if (src.progressive_mode) {
        snprintf(err, errlen, "progressive JPEG: not supported for coefficient-level redaction");
        jpeg_destroy_decompress(&src);
        return 0;
    }
    jvirt_barray_ptr *arrays = jpeg_read_coefficients(&src);

    /* luma: listed blocks -> DC that decodes to 0, all AC 0 */
    jpeg_component_info *y = src.comp_info;
    int q00 = src.quant_tbl_ptrs[y->quant_tbl_no]->quantval[0];
    /* floor(-1024 / q00): level shift 128 -> DC = 8 * (0 - 128) */
    int black = -1024 / q00;
    if (black * q00 != -1024)
        black -= 1;
    for (JDIMENSION r = 0; r < y->height_in_blocks && (int)r < luma_rows; r++) {
        JBLOCKARRAY row = (src.mem->access_virt_barray)((j_common_ptr)&src, arrays[0], r, 1, TRUE);
        for (JDIMENSION c = 0; c < y->width_in_blocks && (int)c < luma_cols; c++) {
            if (!luma[(size_t)r * luma_cols + c])
                continue;
            memset(row[0][c], 0, 64 * sizeof(JCOEF));
            row[0][c][0] = (JCOEF)black;
        }
    }
    /* chroma: listed blocks -> all zero (DC 0 = neutral) */
    for (int comp = 1; comp < src.num_components && comp < 3 && chroma; comp++) {
        jpeg_component_info *ci = src.comp_info + comp;
        for (JDIMENSION r = 0; r < ci->height_in_blocks && (int)r < chroma_rows; r++) {
            JBLOCKARRAY row = (src.mem->access_virt_barray)((j_common_ptr)&src, arrays[comp], r, 1, TRUE);
            for (JDIMENSION c = 0; c < ci->width_in_blocks && (int)c < chroma_cols; c++)
                if (chroma[(size_t)r * chroma_cols + c])
                    memset(row[0][c], 0, 64 * sizeof(JCOEF));
        }
    }

    jpeg_create_compress(&dst);
    dst_created = 1;
    jpeg_mem_dest(&dst, out, outlen);
    jpeg_copy_critical_parameters(&src, &dst);
    dst.optimize_coding = FALSE; /* standard Huffman tables: deterministic output */
    jpeg_write_coefficients(&dst, arrays);
    for (jpeg_saved_marker_ptr m = src.marker_list; m; m = m->next)
        if (keep_marker(m))
            jpeg_write_marker(&dst, m->marker, m->data, m->data_length);
    jpeg_finish_compress(&dst);
    jpeg_destroy_compress(&dst);
    jpeg_finish_decompress(&src);
    jpeg_destroy_decompress(&src);
    return 1;
}

void jdct_free_buffer(unsigned char *p) { free(p); }
