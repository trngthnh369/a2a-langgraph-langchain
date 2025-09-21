# A2A Client - Asynchronous Agent Interaction

This project provides an asynchronous client for interacting with an A2A (Agent-to-Agent) system. It allows you to send messages to an agent, attach files, and manage tasks.

## Description

The `client.py` script implements a command-line interface (CLI) for communicating with an A2A agent.  It uses the `a2a` library to handle communication and data serialization. Key features include:

-   Fetching agent cards.
-   Sending text and file-based messages.
-   Managing task contexts.
-   Retrieving task history.
-   Handling push notifications (optional).

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/trngthnh369/a2a-langgraph-langchain.git
    cd a2a-langgraph-langchain
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    -   On Windows:

        ```bash
        .venv\Scripts\activate
        ```

    -   On macOS and Linux:

        ```bash
        source .venv/bin/activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the client, execute the `client.py` script with the desired options.

```bash
python client.py [options]
```

### Options

*   `--agent`:  The address of the A2A agent (default: `http://localhost:10000`).
*   `--session`:  The session ID (default: `0`).
*   `--history`:  Enable task history retrieval (default: `False`).
*   `--use_push_notifications`: Enable push notifications (default: `False`).
*   `--push_notification_receiver`: The address of the push notification receiver (default: `http://localhost:5000`).

### Example

To start a session with an agent running on `http://localhost:10000`:

```bash
python client.py --agent http://localhost:10000
```

The client will prompt you to enter a message to send to the agent. You can also attach a file.  The agent's response will be displayed.  You can continue interacting with the agent until you type `:q` or `quit`.

### Push Notifications

To use push notifications, you need to:

1.  Set up a push notification receiver.
2.  Enable the `--use_push_notifications` option.
3.  Specify the address of the receiver using the `--push_notification_receiver` option.

Example:

```bash
python client.py --agent http://localhost:10000 --use_push_notifications --push_notification_receiver http://localhost:5000