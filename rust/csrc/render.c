/* Rendering a PDF page with MuPDF, exactly as PyMuPDF's
 * page.get_pixmap(matrix=Matrix(z, z), colorspace=csRGB, alpha=False,
 * annots=True) does: a display list of the page (annotations included),
 * drawn into an RGB pixmap cleared to white. */
#include <mupdf/fitz.h>
#include <mupdf/pdf.h>
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

static fz_document *open_pdf(fz_context *ctx, const unsigned char *data, size_t len, fz_buffer **buf, fz_stream **stm)
{
    fz_register_document_handlers(ctx);
    *buf = fz_new_buffer_from_copied_data(ctx, data, len);
    *stm = fz_open_buffer(ctx, *buf);
    return fz_open_document_with_stream(ctx, "pdf", *stm);
}

/* The bytes of image object `xref`, shown on page `pageno`, as PyMuPDF's
 * Document.extract_image returns them (_make_image_dict): the compressed
 * data of JPEG, JPX, PNG, ... images as stored (CMYK JPEG re-encoded,
 * colours inverted); anything else (CCITT fax, Flate, LZW, JBIG2, raw)
 * decoded and written as PNG. `ext` gets "jpeg", "png", or MuPDF's name of
 * the stored type.
 *
 * The page is loaded and run first, as PyMuPDF has done by then (it lists
 * the page's images and where they are drawn): in a damaged file that is
 * what makes MuPDF repair the cross-reference table, and only after the
 * repair does `xref` name the right object. */
int mimage_extract(const unsigned char *data, size_t len, int pageno, int xref, unsigned char **out, size_t *outlen,
                   char *ext, int extlen, char *err, int errlen)
{
    fz_context *ctx = fz_new_context(NULL, NULL, FZ_STORE_DEFAULT);
    fz_buffer *buf = NULL, *res = NULL;
    fz_stream *stm = NULL;
    fz_document *doc = NULL;
    fz_image *img = NULL;
    pdf_obj *obj = NULL;
    fz_page *page = NULL;
    fz_device *dev = NULL;
    int ok = 0;
    *out = NULL;
    if (!ctx) {
        snprintf(err, errlen, "cannot create MuPDF context");
        return 0;
    }
    fz_var(buf); fz_var(res); fz_var(stm); fz_var(doc); fz_var(img); fz_var(obj); fz_var(page); fz_var(dev);
    fz_try(ctx) {
        doc = open_pdf(ctx, data, len, &buf, &stm);
        pdf_document *pdf = pdf_specifics(ctx, doc);
        if (!pdf)
            fz_throw(ctx, FZ_ERROR_ARGUMENT, "not a PDF");
        page = fz_load_page(ctx, doc, pageno);
        fz_rect bounds = fz_empty_rect;
        dev = fz_new_bbox_device(ctx, &bounds);
        fz_run_page(ctx, page, dev, fz_identity, NULL);
        fz_close_device(ctx, dev);
        obj = pdf_new_indirect(ctx, pdf, xref, 0);
        img = pdf_load_image(ctx, pdf, obj);
        int type = fz_compressed_image_type(ctx, img);
        fz_compressed_buffer *cbuf = fz_compressed_image_buffer(ctx, img);
        if (!cbuf || type == FZ_IMAGE_JBIG2 || type == FZ_IMAGE_UNKNOWN || type < FZ_IMAGE_BMP) {
            res = fz_new_buffer_from_image_as_png(ctx, img, fz_default_color_params);
            snprintf(ext, extlen, "png");
        } else if (type == FZ_IMAGE_JPEG && img->n == 4) {
            res = fz_new_buffer_from_image_as_jpeg(ctx, img, fz_default_color_params, 95, 1);
            snprintf(ext, extlen, "jpeg");
        } else {
            res = fz_keep_buffer(ctx, cbuf->buffer);
            snprintf(ext, extlen, "%s", type == FZ_IMAGE_JPEG ? "jpeg" : fz_image_type_name(type));
        }
        unsigned char *p;
        size_t n = fz_buffer_storage(ctx, res, &p);
        unsigned char *o = malloc(n ? n : 1);
        if (!o)
            fz_throw(ctx, FZ_ERROR_SYSTEM, "out of memory");
        memcpy(o, p, n);
        *out = o;
        *outlen = n;
        ok = 1;
    }
    fz_always(ctx) {
        fz_drop_device(ctx, dev);
        fz_drop_page(ctx, page);
        fz_drop_buffer(ctx, res);
        fz_drop_image(ctx, img);
        pdf_drop_obj(ctx, obj);
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

/* A PDF that only MuPDF can read (e.g. a broken cross-reference table):
 * its page count after MuPDF's repair and, if it has pages, the repaired
 * document written out (objects keep their numbers). */
int mrepair(const unsigned char *data, size_t len, unsigned char **out, size_t *outlen, int *pages,
            char *err, int errlen)
{
    fz_context *ctx = fz_new_context(NULL, NULL, FZ_STORE_DEFAULT);
    fz_buffer *buf = NULL, *res = NULL;
    fz_stream *stm = NULL;
    fz_document *doc = NULL;
    fz_output *o = NULL;
    int ok = 0;
    *out = NULL;
    *outlen = 0;
    if (!ctx) {
        snprintf(err, errlen, "cannot create MuPDF context");
        return 0;
    }
    fz_var(buf); fz_var(res); fz_var(stm); fz_var(doc); fz_var(o);
    fz_try(ctx) {
        doc = open_pdf(ctx, data, len, &buf, &stm);
        pdf_document *pdf = pdf_specifics(ctx, doc);
        if (!pdf)
            fz_throw(ctx, FZ_ERROR_ARGUMENT, "not a PDF");
        *pages = fz_count_pages(ctx, doc);
        if (*pages > 0) {
            res = fz_new_buffer(ctx, len + 1024);
            o = fz_new_output_with_buffer(ctx, res);
            pdf_write_options opts = pdf_default_write_options;
            pdf_write_document(ctx, pdf, o, &opts);
            fz_close_output(ctx, o);
            unsigned char *p;
            size_t n = fz_buffer_storage(ctx, res, &p);
            unsigned char *c = malloc(n ? n : 1);
            if (!c)
                fz_throw(ctx, FZ_ERROR_SYSTEM, "out of memory");
            memcpy(c, p, n);
            *out = c;
            *outlen = n;
        }
        ok = 1;
    }
    fz_always(ctx) {
        fz_drop_output(ctx, o);
        fz_drop_buffer(ctx, res);
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
