/* Rendering a PDF page with MuPDF, exactly as PyMuPDF's
 * page.get_pixmap(matrix=Matrix(z, z), colorspace=csRGB, alpha=False,
 * annots=True) does: a display list of the page (annotations included),
 * drawn into an RGB pixmap cleared to white. */
#include <mupdf/fitz.h>
#include <stdlib.h>
#include <string.h>

int mrender_page(const unsigned char *data, size_t len, int pageno, float zoom,
                 unsigned char **out, int *w, int *h, char *err, int errlen)
{
    fz_context *ctx = fz_new_context(NULL, NULL, FZ_STORE_DEFAULT);
    fz_buffer *buf = NULL;
    fz_stream *stm = NULL;
    fz_document *doc = NULL;
    fz_page *page = NULL;
    fz_display_list *list = NULL;
    fz_pixmap *pix = NULL;
    fz_device *dev = NULL;
    int ok = 0;
    *out = NULL;
    if (!ctx) {
        snprintf(err, errlen, "cannot create MuPDF context");
        return 0;
    }
    fz_var(buf); fz_var(stm); fz_var(doc); fz_var(page); fz_var(list); fz_var(pix); fz_var(dev);
    fz_try(ctx) {
        fz_register_document_handlers(ctx);
        buf = fz_new_buffer_from_copied_data(ctx, data, len);
        stm = fz_open_buffer(ctx, buf);
        doc = fz_open_document_with_stream(ctx, "pdf", stm);
        page = fz_load_page(ctx, doc, pageno);
        list = fz_new_display_list_from_page(ctx, page);
        fz_matrix m = fz_scale(zoom, zoom);
        fz_rect rect = fz_intersect_rect(fz_bound_display_list(ctx, list), fz_infinite_rect);
        fz_irect irect = fz_round_rect(fz_transform_rect(rect, m));
        pix = fz_new_pixmap_with_bbox(ctx, fz_device_rgb(ctx), irect, NULL, 0);
        fz_clear_pixmap_with_value(ctx, pix, 0xFF);
        dev = fz_new_draw_device(ctx, m, pix);
        fz_run_display_list(ctx, list, dev, fz_identity, fz_infinite_rect, NULL);
        fz_close_device(ctx, dev);
        int pw = fz_pixmap_width(ctx, pix), ph = fz_pixmap_height(ctx, pix);
        int n = fz_pixmap_components(ctx, pix), stride = (int)fz_pixmap_stride(ctx, pix);
        unsigned char *s = fz_pixmap_samples(ctx, pix);
        unsigned char *o = malloc((size_t)pw * ph * 3);
        if (!o)
            fz_throw(ctx, FZ_ERROR_SYSTEM, "out of memory");
        for (int y = 0; y < ph; y++)
            for (int x = 0; x < pw; x++)
                for (int c = 0; c < 3; c++)
                    o[((size_t)y * pw + x) * 3 + c] = s[(size_t)y * stride + (size_t)x * n + (c < n ? c : n - 1)];
        *out = o;
        *w = pw;
        *h = ph;
        ok = 1;
    }
    fz_always(ctx) {
        fz_drop_device(ctx, dev);
        fz_drop_pixmap(ctx, pix);
        fz_drop_display_list(ctx, list);
        fz_drop_page(ctx, page);
        fz_drop_document(ctx, doc);
        fz_drop_stream(ctx, stm);
        fz_drop_buffer(ctx, buf);
    }
    fz_catch(ctx) {
        snprintf(err, errlen, "%s", fz_caught_message(ctx));
        ok = 0;
    }
    fz_drop_context(ctx);
    return ok;
}

void mrender_free(unsigned char *p) { free(p); }

/* MuPDF's version, for the record of how a page was rendered. */
const char *mrender_version(void) { return FZ_VERSION; }
