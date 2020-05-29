import mock
from listen import run_lungmask
import os



@mock.patch('SimpleITK.GetArrayFromImage')
@mock.patch('numpy.save')
@mock.patch('time.time')
@mock.patch('lungmask.lungmask.get_model')
@mock.patch('lungmask.lungmask.apply')
@mock.patch('lungmask.utils.get_input_image')
def test_run_lungmask_calls_time_once(mock_lungmask_get_input_image, mock_lungmask_apply,
                                                    mock_lungmask_get_model, mock_time, mock_numpy_save,
                                                    mock_sitk_get_array_from_image):

    def mock_exists(*args, **kwargs):
        return True
    os.path.exists = mock_exists

    class MockInputImage():
        def GetSpacing(self):
            return (1, 2, 3)

    mock_time.return_value = "123.123"
    mock_lungmask_get_model.return_value = "MockModel123"
    mock_lungmask_get_input_image.return_value = MockInputImage()

    param_dict = {
        "source_dir": ["/source/dir"],
        "model_name": ["mock model name"]
    }

    run_lungmask(param_dict)
    mock_time.assert_called_once()


@mock.patch('SimpleITK.GetArrayFromImage')
@mock.patch('numpy.save')
@mock.patch('time.time')
@mock.patch('lungmask.lungmask.get_model')
@mock.patch('lungmask.lungmask.apply')
@mock.patch('lungmask.utils.get_input_image')
def test_run_lungmask_calls_get_model_once_with_name_from_param_dict(mock_lungmask_get_input_image, mock_lungmask_apply,
                                                                     mock_lungmask_get_model, mock_time, mock_np_save,
                                                                     mock_sitk_get_array_from_image):

    def mock_exists(*args, **kwargs):
        return True
    os.path.exists = mock_exists

    class MockInputImage():
        def GetSpacing(self):
            return (1, 2, 3)

    mock_time.return_value = "123.123"
    mock_lungmask_get_model.return_value = "MockModel123"
    mock_lungmask_get_input_image.return_value = MockInputImage()

    mock_model_name = "mock model name"
    param_dict = {
        "source_dir": ["/source/dir"],
        "model_name": [mock_model_name]
    }

    run_lungmask(param_dict)
    mock_lungmask_get_model.assert_called_once_with('unet', mock_model_name)


@mock.patch('SimpleITK.GetArrayFromImage')
@mock.patch('numpy.save')
@mock.patch('time.time')
@mock.patch('lungmask.lungmask.get_model')
@mock.patch('lungmask.lungmask.apply')
@mock.patch('lungmask.utils.get_input_image')
def test_run_lungmask_calls_apply_once_with_correct_parameters(mock_lungmask_get_input_image, mock_lungmask_apply,
                                                               mock_lungmask_get_model, mock_time, mock_np_save,
                                                               mock_sitk_get_array_from_image):

    def mock_exists(*args, **kwargs):
        return True
    os.path.exists = mock_exists

    class MockInputImage():
        def GetSpacing(self):
            return (1, 2, 3)

    class MockModel():
        pass

    mock_model = MockModel()
    mock_input_image = MockInputImage()

    mock_time.return_value = "123.123"
    mock_lungmask_get_model.return_value = mock_model
    mock_lungmask_get_input_image.return_value = mock_input_image

    mock_model_name = "mock model name"
    param_dict = {
        "source_dir": ["/source/dir"],
        "model_name": [mock_model_name]
    }

    run_lungmask(param_dict)
    mock_lungmask_apply.assert_called_once_with(mock_input_image, mock_model,
                                                force_cpu=False, batch_size=20, volume_postprocessing=False)


@mock.patch('SimpleITK.GetArrayFromImage')
@mock.patch('numpy.save')
@mock.patch('time.time')
@mock.patch('lungmask.lungmask.get_model')
@mock.patch('lungmask.lungmask.apply')
@mock.patch('lungmask.utils.get_input_image')
def test_run_lungmask_calls_numpy_save_twice(mock_lungmask_get_input_image, mock_lungmask_apply,
                                             mock_lungmask_get_model, mock_time, mock_np_save,
                                             mock_sitk_get_array_from_image):

    def mock_exists(*args, **kwargs):
        return True
    os.path.exists = mock_exists

    class MockInputImage():
        def GetSpacing(self):
            return (1, 2, 3)

    class MockModel():
        pass

    mock_model = MockModel()
    mock_input_image = MockInputImage()

    mock_time.return_value = "123.123"
    mock_lungmask_get_model.return_value = mock_model
    mock_lungmask_get_input_image.return_value = mock_input_image

    mock_model_name = "mock model name"
    param_dict = {
        "source_dir": ["/source/dir"],
        "model_name": [mock_model_name]
    }

    run_lungmask(param_dict)
    assert mock_np_save.call_count == 2


@mock.patch('SimpleITK.GetArrayFromImage')
@mock.patch('numpy.save')
@mock.patch('time.time')
@mock.patch('lungmask.lungmask.get_model')
@mock.patch('lungmask.lungmask.apply')
@mock.patch('lungmask.utils.get_input_image')
def test_run_lungmask_calls_sitk_get_array_from_image_once(mock_lungmask_get_input_image, mock_lungmask_apply,
                                                           mock_lungmask_get_model, mock_time, mock_np_save,
                                                           mock_sitk_get_array_from_image):

    def mock_exists(*args, **kwargs):
        return True
    os.path.exists = mock_exists

    class MockInputImage():
        def GetSpacing(self):
            return (1, 2, 3)

    class MockModel():
        pass

    mock_model = MockModel()
    mock_input_image = MockInputImage()

    mock_time.return_value = "123.123"
    mock_lungmask_get_model.return_value = mock_model
    mock_lungmask_get_input_image.return_value = mock_input_image

    mock_model_name = "mock model name"
    param_dict = {
        "source_dir": ["/source/dir"],
        "model_name": [mock_model_name]
    }

    run_lungmask(param_dict)
    mock_sitk_get_array_from_image.assert_called_once()


@mock.patch('SimpleITK.GetArrayFromImage')
@mock.patch('numpy.save')
@mock.patch('time.time')
@mock.patch('lungmask.lungmask.get_model')
@mock.patch('lungmask.lungmask.apply')
@mock.patch('lungmask.utils.get_input_image')
def test_run_lungmask_sends_success_false_when_input_image_doesnt_exist(mock_lungmask_get_input_image,
                                                                        mock_lungmask_apply,
                                                                        mock_lungmask_get_model, mock_time,
                                                                        mock_np_save,
                                                                        mock_sitk_get_array_from_image):
    def mock_exists_false(*args, **kwargs):
        return False

    os.path.exists = mock_exists_false

    class MockInputImage():
        def GetSpacing(self):
            return (1, 2, 3)

    class MockModel():
        pass

    mock_model = MockModel()
    mock_input_image = MockInputImage()

    mock_time.return_value = "123.123"
    mock_lungmask_get_model.return_value = mock_model
    mock_lungmask_get_input_image.return_value = mock_input_image

    mock_model_name = "mock model name"
    param_dict = {
        "source_dir": ["/source/dir"],
        "model_name": [mock_model_name]
    }

    result_dict, success = run_lungmask(param_dict)

    assert not success

@mock.patch('SimpleITK.GetArrayFromImage')
@mock.patch('numpy.save')
@mock.patch('time.time')
@mock.patch('lungmask.lungmask.get_model')
@mock.patch('lungmask.lungmask.apply')
@mock.patch('lungmask.utils.get_input_image')
def test_run_lungmask_result_dictionary_contains_segmentation_and_input(mock_lungmask_get_input_image,
                                                                        mock_lungmask_apply,
                                                                        mock_lungmask_get_model, mock_time,
                                                                        mock_np_save,
                                                                        mock_sitk_get_array_from_image):
    def mock_exists(*args, **kwargs):
        return True
    os.path.exists = mock_exists

    class MockInputImage():
        def GetSpacing(self):
            return (1, 2, 3)

    class MockModel():
        pass

    mock_model = MockModel()
    mock_input_image = MockInputImage()

    mock_time.return_value = "123.123"
    mock_lungmask_get_model.return_value = mock_model
    mock_lungmask_get_input_image.return_value = mock_input_image

    mock_model_name = "mock model name"
    param_dict = {
        "source_dir": ["/source/dir"],
        "model_name": [mock_model_name]
    }

    result_dict, success = run_lungmask(param_dict)

    assert success
    assert "segmentation" in result_dict
    assert "input" in result_dict