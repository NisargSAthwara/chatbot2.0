import React from 'react';

interface SearchInfo {
    stages: string[];
    query: string;
    urls: string[];
}

interface Message {
    id: number;
    content: string;
    isUser: boolean;
    type: string;
    isLoading?: boolean;
    searchInfo?: SearchInfo;
}

interface MessageAreaProps {
    messages: Message[];
}

interface SearchStagesProps {
    searchInfo: SearchInfo;
}

const PremiumTypingAnimation: React.FC = () => {
    return (
        <div className="flex space-x-2">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
        </div>
    );
};

const SearchStages: React.FC<SearchStagesProps> = ({ searchInfo }) => {
    return (
        <div className="mb-2 text-sm text-gray-600">
            {searchInfo.stages.map((stage, index) => (
                <div key={index} className="mb-1">{stage}</div>
            ))}
            {searchInfo.urls && searchInfo.urls.map((url, index) => (
                <div key={index} className="bg-gray-100 text-xs px-3 py-1.5 rounded border border-gray-200 truncate max-w-[200px]">
                    {String(url)}
                </div>
            ))}
        </div>
    );
};

const MessageArea: React.FC<MessageAreaProps> = ({ messages }) => {
    return (
        <div className="flex-grow overflow-y-auto bg-[#FCFCF8] border-b border-gray-100" style={{ minHeight: 0 }}>
            <div className="max-w-4xl mx-auto p-6">
                {messages.map((message) => (
                    <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-5`}>
                        <div className="flex flex-col max-w-md">
                            {!message.isUser && message.searchInfo && (
                                <SearchStages searchInfo={message.searchInfo} />
                            )}
                            <div
                                className={`rounded-lg py-3 px-5 ${message.isUser
                                    ? 'bg-gradient-to-br from-[#5E507F] to-[#4A3F71] text-white rounded-br-none shadow-md'
                                    : 'bg-[#F3F3EE] text-gray-800 border border-gray-200 rounded-bl-none shadow-sm'
                                    }`}
                            >
                                {message.isLoading ? (
                                    <PremiumTypingAnimation />
                                ) : (
                                    message.content || (
                                        <span className="text-gray-400 text-xs italic">Waiting for response...</span>
                                    )
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default MessageArea;