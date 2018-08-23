"""Example tools made from popular libraries for parsing / importing a variety of image formats."""
from io import BytesIO
import tempfile
import os
from PIL import Image
import cv2
from wand.image import Image as WandIm
from wand.image import Color as WandCol
from PyPDF2 import PdfFileReader, PdfFileWriter


#____PDF_CONSTANTS____#
LOW_DENSITY_DEFAULT_VALUE = 200
HIGH_DENSITY_DEFAULT_VALUE = 400

THUMBNAIL_WIDTH_DEFAULT = 500
THUMBNAIL_HEIGHT_DEFAULT = 300

DEFAULT_OUT_FORMAT = 'png'
DEFAULT_IMAGE_LIB = 'PILIM'
DEFAULT_ACCEPTED_IMG_FORMATS = {'.png', '.jpg', 'jpeg', '.gif'}  # 4 character lengths
DEFAULT_PDF_FORMATS = {'.pdf'}


def create_pil_thumbnail_dict_from_path(file_alias_and_name_tup, *,
                                        thumbnail_width=THUMBNAIL_WIDTH_DEFAULT,
                                        thumbnail_height=THUMBNAIL_HEIGHT_DEFAULT,
                                        out_format=DEFAULT_OUT_FORMAT,
                                        accepted_image_formats=DEFAULT_ACCEPTED_IMG_FORMATS,
                                        accepted_pdf_formats=DEFAULT_PDF_FORMATS):
    """
    Import image or pdf and create a thumbnail in the Pillow library format, made for batch Tkinter use with Pool.
    :Parameters:
        :param tuple file_alias_and_name_tup: tuple containing the return dict key-name and the document file path.
        :param int thumbnail_width: thumbnail width (default is 500)
        :param int thumbnail_height: thumbnail height (default is 300)
        :param str out_format: set the pdf image magick image format used during the pdf-to-image conversion.
        :param dict of strings accepted_image_formats: determines accepted image file endings.
        :param dict of strings accepted_pdf_formats: determines accepted pdf endings.
    :rtype: Dictionary of Dict
    :return dict return_dict: {<file_alias>: {"path": <filename>, "thumbnail": <Pillow Image Object>}}
        file_alias is file_alias_and_name_tup input index 0, filename is file_alias_and_name input index 1.
        Pillow Image Object is the imported image/pdf thumbnail, or None.
    """
    def _convert_pdf_object_with_attributes(pdf_object):
        pdf_object.format = out_format
        pdf_object.background_color = WandCol('white')
        pdf_object.alpha_channel = 'remove'
        pdf_object.resize(thumbnail_width, thumbnail_height)
        image_data = BytesIO()
        pdf_object.save(file=image_data)
        image_data.seek(0)
        converted_image = Image.open(image_data)
        return converted_image

    file_alias, filename = file_alias_and_name_tup
    ending = filename[-4:]
    if ending in accepted_pdf_formats:
        image_in = WandIm(filename=filename)
        pages = len(image_in.sequence)
        if pages > 1:
            h = image_in.height * pages
            w = image_in.width
            img = WandIm(width=w, height=h)
            for i in range(pages):
                img.composite(image_in.sequence[i],
                              top=image_in.height * i,
                              left=0)
            pilim = _convert_pdf_object_with_attributes(img)
        else:
            pilim = _convert_pdf_object_with_attributes(image_in)
    elif ending in accepted_image_formats:
        img = Image.open(filename)
        pilim = img.resize((thumbnail_width, thumbnail_height), Image.ANTIALIAS)
    else:
        pilim = None
    return_dict = {file_alias: {"path": filename, "thumbnail": pilim}}
    return return_dict


def import_any_image_file(image_file_path, im_mode=None, *,
                          out_resolution=LOW_DENSITY_DEFAULT_VALUE,
                          default_format=DEFAULT_OUT_FORMAT,
                          accepted_image_formats=DEFAULT_ACCEPTED_IMG_FORMATS,
                          accepted_pdf_formats=DEFAULT_PDF_FORMATS):
    """
    Import any image format and return an image object from the library of your choosing.
    :Parameters:
        :param str image_file_path: input path for document to be converted to image.
        :param str im_mode: determine the returned image object's library.
            PILIM will return a Pillow image,
            CVGRAY will return a grayscale OpenCV image,
            Any other input will return a default OpenCV image.
        :param int out_resolution: set the pdf image magick density used during the pdf-to-image conversion.
        :param str default_format: set the pdf image magick image format used during the pdf-to-image conversion.
        :param dict of strings accepted_image_formats: determines accepted image file endings.
        :param dict of strings accepted_pdf_formats: determines accepted pdf endings.
    :rtype: Python Image Object, or None
    :return python object out_im: imported image of library format im_mode or None.
    Notes: Raises IOError for read errors, and prints warning if temp-file removal failed.
    PDF background color is hard-coded to white due to standard, feel free to change/add parameter.
    """
    def _import_pdf():
        def _convert_single_pdf_page(page):
            page.format = default_format
            page.background_color = WandCol('white')
            page.alpha_channel = 'remove'
            tempf = tempfile.NamedTemporaryFile(suffix='.' + default_format,
                                                prefix="pdf_convert",
                                                delete=False)
            page.save(filename=tempf.name)
            tempf.close()
            if im_mode == "PILIM":
                this_out_im = Image.open(tempf.name)
                this_out_im.load()
            elif im_mode == "CVGRAY":
                temp_out_im = cv2.imread(tempf.name, cv2.IMREAD_GRAYSCALE)
                this_out_im = temp_out_im.copy()
                del temp_out_im
            else:
                temp_out_im = cv2.imread(tempf.name)
                this_out_im = temp_out_im.copy()
                del temp_out_im
            tempf.close()
            os.remove(tempf.name)
            if os.path.isfile(tempf.name):
                tempf.close()
                os.remove(tempf.name)
                if os.path.isfile(tempf.name):
                    print("TEMP FILE REMOVAL FAILED")
                else:
                    tempf = None
            else:
                tempf = None
            return this_out_im, tempf

        pdf_out_im = []
        tempf = None
        try:  # Note: Modified wand.image.Image.destroy() to no longer raise TypeError.
            with WandIm(filename=image_file_path, resolution=out_resolution) as image_in:
                pages = len(image_in.sequence)
                if pages > 1:
                    for i in range(pages):
                        with WandIm(image_in.sequence[i]) as img:
                            try:
                                this_converted_img, tempf = _convert_single_pdf_page(image_in)
                            except IOError:
                                raise
                            else:
                                pdf_out_im.append(this_converted_img)
                            finally:
                                if tempf:
                                    if os.path.isfile(tempf.name):
                                        tempf.close()
                                        os.remove(tempf.name)
                                        if os.path.isfile(tempf.name):
                                            print("REPEATED TEMP FILE REMOVAL FAILED")
                else:
                    pdf_out_im, tempf = _convert_single_pdf_page(image_in)
        except IOError:
            raise
        finally:
            if tempf:
                if os.path.isfile(tempf.name):
                    tempf.close()
                    os.remove(tempf.name)
                    if os.path.isfile(tempf.name):
                        print("FINAL TEMP FILE REMOVAL FAILED")
                    tempf = None
        return pdf_out_im

    ending = image_file_path[-4:]
    if ending in accepted_pdf_formats:
        out_im = _import_pdf()
    elif ending in accepted_image_formats:
        if im_mode == "PILIM":
            out_im = Image.open(image_file_path)
        elif im_mode == "CVGRAY":
            out_im = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
        else:
            out_im = cv2.imread(image_file_path)
    else:
        out_im = None
    return out_im
