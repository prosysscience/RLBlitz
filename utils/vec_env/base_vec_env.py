import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
from collections import OrderedDict

import cloudpickle
import gym
import numpy as np

# Define type aliases here to avoid circular import
# Used when we want to access one or more VecEnv
VecEnvIndices = Union[None, int, Iterable[int]]
# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = Tuple[VecEnvObs, np.ndarray, np.ndarray, List[Dict]]


def tile_images(img_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    :param img_nhwc: list or array of images, ndim=4 once turned into array. img nhwc
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image


class StableBaselinesVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.

    :param num_envs: the number of environments
    :param observation_space: the observation space
    :param action_space: the action space
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs: int, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation
        """
        raise NotImplementedError()

    @abstractmethod
    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """
        Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        raise NotImplementedError()

    @abstractmethod
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """
        Set attribute inside vectorized environments.

        :param attr_name: The name of attribute to assign new value
        :param value: Value to assign to `attr_name`
        :param indices: Indices of envs to assign value
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """
        Call instance methods of vectorized environments.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: List of items returned by the environment's method call
        """
        raise NotImplementedError()

    @abstractmethod
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        raise NotImplementedError()

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def get_images(self) -> Sequence[np.ndarray]:
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering

        :param mode: the rendering type
        """
        try:
            imgs = self.get_images()
        except NotImplementedError:
            print(f"Render not defined for {self}")
            return

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == "human":
            import cv2  # pytype:disable=import-error

            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")

    @abstractmethod
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        pass

    @property
    def unwrapped(self) -> "StableBaselinesVecEnv":
        if isinstance(self, StableBaselinesVecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name: name of attribute to check for
        :param already_found: whether this attribute has already been found in a wrapper
        :return: name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return f"{type(self).__module__}.{type(self).__name__}"
        else:
            return None

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


class StableBaselinesVecEnvWrapper(StableBaselinesVecEnv):
    """
    Vectorized environment base class

    :param venv: the vectorized environment to wrap
    :param observation_space: the observation space (can be None to load from venv)
    :param action_space: the action space (can be None to load from venv)
    """

    def __init__(
        self,
        venv: StableBaselinesVecEnv,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
    ):
        self.venv = venv
        StableBaselinesVecEnv.__init__(
            self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space,
        )
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self) -> VecEnvObs:
        pass

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.venv.seed(seed)

    def close(self) -> None:
        return self.venv.close()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.venv.render(mode=mode)

    def get_images(self) -> Sequence[np.ndarray]:
        return self.venv.get_images()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return self.venv.env_is_wrapped(wrapper_class, indices=indices)

    def __getattr__(self, name: str) -> Any:
        """Find attribute from wrapped venv(s) if this wrapper does not have it.
        Useful for accessing attributes from venvs which are wrapped with multiple wrappers
        which have unique attributes of interest.
        """
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = f"{type(self).__module__}.{type(self).__name__}"
            error_str = (
                f"Error: Recursive attribute lookup for {name} from {own_class} is "
                "ambiguous and hides attribute from {blocked_class}"
            )
            raise AttributeError(error_str)

        return self.getattr_recursive(name)

    def _get_all_attributes(self) -> Dict[str, Any]:
        """Get all (inherited) instance and class attributes

        :return: all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name: str) -> Any:
        """Recursively check wrappers to find attribute.

        :param name: name of attribute to look for
        :return: attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.venv, "getattr_recursive"):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.venv.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.venv, name)

        return attr

    def getattr_depth_check(self, name: str, already_found: bool) -> str:
        """See base class.

        :return: name of module whose attribute is being shadowed, if any.
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes and already_found:
            # this venv's attribute is being hidden because of a higher venv.
            shadowed_wrapper_class = f"{type(self).__module__}.{type(self).__name__}"
        elif name in all_attributes and not already_found:
            # we have found the first reference to the attribute. Now check for duplicates.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, True)
        else:
            # this wrapper does not have the attribute. Keep searching.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, already_found)

        return shadowed_wrapper_class


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


def copy_obs_dict(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Deep-copy a dict of numpy arrays.

    :param obs: a dict of numpy arrays.
    :return: a dict of copied numpy arrays.
    """
    assert isinstance(obs, OrderedDict), f"unexpected type for observations '{type(obs)}'"
    return OrderedDict([(k, np.copy(v)) for k, v in obs.items()])


def dict_to_obs(space: gym.spaces.Space, obs_dict: Dict[Any, np.ndarray]) -> VecEnvObs:
    """
    Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param space: an observation space.
    :param obs_dict: a dict of numpy arrays.
    :return: returns an observation of the same type as space.
        If space is Dict, function is identity; if space is Tuple, converts dict to Tuple;
        otherwise, space is unstructured and returns the value raw_obs[None].
    """
    if isinstance(space, gym.spaces.Dict):
        return obs_dict
    elif isinstance(space, gym.spaces.Tuple):
        assert len(obs_dict) == len(space.spaces), "size of observation does not match size of observation space"
        return tuple((obs_dict[i] for i in range(len(space.spaces))))
    else:
        assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
        return obs_dict[None]


def obs_space_info(obs_space: gym.spaces.Space) -> Tuple[List[str], Dict[Any, Tuple[int, ...]], Dict[Any, np.dtype]]:
    """
    Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: an observation space
    :return: A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        subspaces = obs_space.spaces
    elif isinstance(obs_space, gym.spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}
    else:
        assert not hasattr(obs_space, "spaces"), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes
