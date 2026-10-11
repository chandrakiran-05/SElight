import pytest
from scripts.import_external_datasets import voc_boxes
from scripts.external_audit import audit_external


def test_voc_inclusive_edges_preserve_full_image_box():
    xml = b'<annotation><size><width>20</width><height>10</height></size><object><name>person</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>10</ymax></bndbox></object></annotation>'
    assert voc_boxes(xml,20,10) == [[0,0,0,20,10]]
    with pytest.raises(ValueError,match='dimensions'):
        voc_boxes(xml,21,10)


def test_external_does_not_score_unannotated_classes():
    with pytest.raises(ValueError,match='person evaluation'):
        audit_external({'kind':'external_transfer','evaluation_classes':[0,1]}, {})
