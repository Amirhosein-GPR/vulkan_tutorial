use ash::{vk, LoadingError};
use png::DecodingError;
use std::io;
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

impl From<SuitabilityError> for ApplicationError {
    fn from(value: SuitabilityError) -> Self {
        Self::EngineError(format!("Suitability error: {}", value.0))
    }
}

impl From<(Vec<vk::Pipeline>, vk::Result)> for ApplicationError {
    fn from(value: (Vec<vk::Pipeline>, vk::Result)) -> Self {
        Self::VulkanError(value.1)
    }
}

impl From<io::Error> for ApplicationError {
    fn from(value: io::Error) -> Self {
        Self::EngineError(value.to_string())
    }
}

impl From<DecodingError> for ApplicationError {
    fn from(value: DecodingError) -> Self {
        Self::EngineError(value.to_string())
    }
}

#[derive(Debug)]
pub struct SuitabilityError(pub String);

impl Display for SuitabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for SuitabilityError {}
