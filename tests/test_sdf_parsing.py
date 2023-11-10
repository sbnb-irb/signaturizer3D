import pytest

from signaturizer3d.data.sdf import gather_sdf_data, parse_sdf


@pytest.fixture
def valid_sdf(tmp_path):
    sdf_content = """
CT1001214542


  9  8  0  0  0               999 V2000
   -0.0173    1.4248    0.0099 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0021   -0.0041    0.0020 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.4181    1.9544   -0.0010 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5445    1.7859   -0.8732 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5275    1.7763    0.9067 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8759   -0.4094    0.0082 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9453    1.5933    0.8821 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9283    1.6028   -0.8978 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4033    3.0442    0.0050 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  2  6  1  0  0  0  0
  3  7  1  0  0  0  0
  3  8  1  0  0  0  0
  3  9  1  0  0  0  0
M  END
$$$$
"""
    sdf_file = tmp_path / "valid.sdf"
    with open(sdf_file, "w") as file:
        file.write(sdf_content.strip())
    return str(sdf_file)


def test_valid_sdf_file(valid_sdf):
    result = parse_sdf(valid_sdf)

    assert len(result) == 1

    atoms, coordinates = result[0]
    expected_atoms = ["C", "O", "C"]
    expected_coordinates = [
        [-0.0173, 1.4248, 0.0099],
        [0.0021, -0.0041, 0.0020],
        [1.4181, 1.9544, -0.0010],
    ]
    assert atoms == expected_atoms
    assert coordinates == expected_coordinates


@pytest.fixture
def malformed_sdf(tmp_path):
    sdf_content = """
    malformed_molecule
     RDKit
     badly formatted sdf content
    """
    sdf_file = tmp_path / "malformed.sdf"
    with open(sdf_file, "w") as file:
        file.write(sdf_content.strip())
    return str(sdf_file)


def test_malformed_sdf_file(malformed_sdf, mocker):
    mocked_logger = mocker.patch("signaturizer3d.data.sdf.logger")

    result = parse_sdf(malformed_sdf)

    assert len(result) == 0
    mocked_logger.warning.assert_called_once_with(
        f"Unable to read molecule from {malformed_sdf}"
    )


@pytest.fixture
def empty_sdf(tmp_path):
    sdf_file = tmp_path / "empty.sdf"
    sdf_file.touch()  # Creates an empty file
    return str(sdf_file)


def test_empty_sdf_file(empty_sdf):
    result = parse_sdf(empty_sdf)

    assert len(result) == 0


@pytest.fixture
def multi_molecule_sdf(tmp_path):
    sdf_content = """
CT1001214542


  9  8  0  0  0               999 V2000
   -0.0173    1.4248    0.0099 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0021   -0.0041    0.0020 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.4181    1.9544   -0.0010 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5445    1.7859   -0.8732 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5275    1.7763    0.9067 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8759   -0.4094    0.0082 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9453    1.5933    0.8821 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9283    1.6028   -0.8978 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4033    3.0442    0.0050 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  2  6  1  0  0  0  0
  3  7  1  0  0  0  0
  3  8  1  0  0  0  0
  3  9  1  0  0  0  0
M  END
$$$$
CT1001789336


  5  4  0  0  0               999 V2000
    0.0021   -0.0041    0.0020 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0127    1.0858    0.0080 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0099    1.4631    0.0003 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5399    1.4469   -0.8751 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5229    1.4373    0.9048 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
  2  5  1  0  0  0  0
M  END
$$$$
    """
    sdf_file = tmp_path / "multi_molecule.sdf"
    with open(sdf_file, "w") as file:
        file.write(sdf_content.strip())
    return str(sdf_file)


def test_multi_molecule_sdf_file(multi_molecule_sdf):
    result = parse_sdf(multi_molecule_sdf)

    atoms, coordinates = result[0]
    expected_atoms = ["C", "O", "C"]
    expected_coordinates = [
        [-0.0173, 1.4248, 0.0099],
        [0.0021, -0.0041, 0.0020],
        [1.4181, 1.9544, -0.0010],
    ]

    assert atoms == expected_atoms
    assert coordinates == expected_coordinates
    atoms, coordinates = result[1]
    expected_atoms = ["C"]
    expected_coordinates = [[-0.0127, 1.0858, 0.0080]]
    assert atoms == expected_atoms
    assert coordinates == expected_coordinates


def test_nonexistent_sdf_file(mocker):
    mocked_logger = mocker.patch("signaturizer3d.data.sdf.logger")

    result = parse_sdf("nonexistent.sdf")

    assert len(result) == 0
    mocked_logger.error.assert_called_once_with("Unable to read file nonexistent.sdf")


sample_molecules = [
    (["C", "H", "O"], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    (["N", "C"], [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
]


@pytest.fixture
def mock_sdf_file(tmp_path):
    sdf_file = tmp_path / "molecule.sdf"
    sdf_file.write_text("...")  # Dummy content
    return str(sdf_file)


@pytest.fixture
def mock_sdf_directory(tmp_path):
    for i in range(3):
        (tmp_path / f"molecule_{i}.sdf").write_text("...")
    return str(tmp_path)


def test_gather_from_single_file(mocker, mock_sdf_file):
    mocked_parse = mocker.patch(
        "signaturizer3d.data.sdf.parse_sdf", return_value=sample_molecules
    )

    atoms, coords = gather_sdf_data(mock_sdf_file)

    assert len(atoms) == 2
    assert len(coords) == 2
    mocked_parse.assert_called_once_with(mock_sdf_file)


def test_gather_from_directory(mocker, mock_sdf_directory):
    mocked_parse = mocker.patch(
        "signaturizer3d.data.sdf.parse_sdf", return_value=sample_molecules
    )
    atoms, coords = gather_sdf_data(mock_sdf_directory)
    # Assuming 3 files in the directory and each file returns the sample molecules
    assert len(atoms) == 6
    assert len(coords) == 6
    assert mocked_parse.call_count == 3


def test_file_not_found_error(mocker):
    mocker.patch("signaturizer3d.data.sdf.parse_sdf", return_value=sample_molecules)
    with pytest.raises(FileNotFoundError):
        gather_sdf_data("/non/existent/path")
