use ash::{vk, LoadingError};
use std::{error::Error, fmt::Display};
use winit::error::OsError;

// All errors in this application are one of these variants, or can be transfered to one of these.
#[derive(Debug)]
pub enum ApplicationError {
    EngineError(String),
    VulkanError(vk::Result),
}

impl Display for ApplicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Error for ApplicationError {}

impl From<OsError> for ApplicationError {
    fn from(value: winit::error::OsError) -> Self {
        Self::EngineError(value.to_string())
    }
}

impl From<vk::Result> for ApplicationError {
    fn from(value: vk::Result) -> Self {
        Self::VulkanError(value)
    }
}

impl From<LoadingError> for ApplicationError {
    fn from(value: ash::LoadingError) -> Self {
        Self::EngineError(value.to_string())
    }
}